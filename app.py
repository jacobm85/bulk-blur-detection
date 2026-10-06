import eventlet
eventlet.monkey_patch()  # Must run before other imports

import csv
import io
import os
import sys

from eventlet import tpool
from flask import Flask, abort, jsonify, render_template, request, send_file
from flask_socketio import SocketIO, emit
from PIL import Image, ImageOps

import library
import scoring

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')

# Use the real (non-green) subprocess module; its output is read in a native thread via tpool
# so the server keeps responding while a job runs, on every platform.
subprocess = eventlet.patcher.original('subprocess')

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.realpath(os.environ.get('IMAGES_DIR', '/app/images'))
DB_PATH = os.environ.get('DB_PATH', os.path.join(APP_DIR, 'data', 'blurdetector.db'))
PAGE_SIZE = 60

current_job = None


def resolve(rel):
    """Absolute path for a path relative to BASE_DIR, or None if it points outside it."""
    full = os.path.realpath(os.path.join(BASE_DIR, (rel or '').lstrip('/')))
    if os.path.commonpath([full, BASE_DIR]) != BASE_DIR:
        return None
    return full


def rel_folder(rel):
    full = resolve(rel)
    if full is None or not os.path.isdir(full):
        abort(400, 'Folder does not exist')
    return library.to_rel(BASE_DIR, full)


def db():
    return library.connect(DB_PATH)


@app.route('/')
def index():
    return render_template('index.html', default_threshold=scoring.DEFAULT_THRESHOLD,
                           default_eye_threshold=scoring.DEFAULT_EYE_THRESHOLD, page_size=PAGE_SIZE)


@socketio.on('browse')
def browse(data):
    path = resolve(data.get('path'))
    if path is None or not os.path.isdir(path):
        emit('error', {'message': 'Directory does not exist!'})
        return
    entries = sorted(os.listdir(path))
    dirs = [e for e in entries if os.path.isdir(os.path.join(path, e)) and not library.skip_dir(e)]
    images = sum(1 for e in entries if scoring.is_image(e))
    emit('files', {'path': library.to_rel(BASE_DIR, path), 'dirs': dirs, 'images': images})


@socketio.on('scan')
def start_scan(data):
    global current_job
    if current_job is not None:
        emit('error', {'message': 'A scan is already running.'})
        return
    path = resolve(data.get('folder'))
    if path is None or not os.path.isdir(path):
        emit('error', {'message': 'Folder does not exist!'})
        return
    command = [sys.executable, '-u', os.path.join(APP_DIR, 'library.py'), 'scan',
               '--root', BASE_DIR, '--folder', library.to_rel(BASE_DIR, path), '--db', DB_PATH]
    current_job = socketio.start_background_task(run_job, command)
    emit('started', {})


def run_job(command):
    global current_job
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in iter(lambda: tpool.execute(process.stdout.readline), ''):
            line = line.rstrip()
            print(line)
            socketio.emit('progress', {'line': line})
        tpool.execute(process.wait)
        if process.returncode == 0:
            socketio.emit('done', {'message': 'Scan completed'})
        else:
            socketio.emit('done', {'message': f'Scan failed (exit code {process.returncode})', 'failed': True})
    finally:
        current_job = None


@socketio.on('connect')
def on_connect():
    emit('status', {'running': current_job is not None})


@app.route('/api/stats')
def api_stats():
    conn = db()
    labelled = conn.execute("SELECT COUNT(*), SUM(label = 'blurry') FROM images WHERE label IS NOT NULL").fetchone()
    return jsonify({**library.stats(conn, rel_folder(request.args.get('folder'))),
                    'labelled': labelled[0], 'labelled_blurry': labelled[1] or 0,
                    'drafts': library.draft_count(conn),
                    'recommendations': library.recommend_thresholds(conn),
                    'eye_recommendation': library.recommend_eye_threshold(conn),
                    'last_batch': library.last_batch(conn)})


@app.route('/api/candidates')
def api_candidates():
    try:
        threshold = float(request.args.get('threshold', scoring.DEFAULT_THRESHOLD))
        eye_threshold = float(request.args.get('eye_threshold', scoring.DEFAULT_EYE_THRESHOLD))
        offset = max(int(request.args.get('offset', 0)), 0)
    except ValueError:
        abort(400, 'Invalid threshold or offset')
    conn = db()
    random_sample = request.args.get('mode') == 'random'
    per_camera = None
    if request.args.get('per_camera') == '1' and not random_sample:
        per_camera = {c: r['threshold'] for c, r in library.recommend_thresholds(conn).items()}
        eye_threshold = (library.recommend_eye_threshold(conn) or {}).get('threshold', eye_threshold)
    total, rows = library.candidates(conn, rel_folder(request.args.get('folder')),
                                     None if random_sample else threshold,
                                     request.args.get('camera') or None,
                                     include_reviewed=request.args.get('reviewed') == '1',
                                     limit=PAGE_SIZE, offset=offset, camera_thresholds=per_camera,
                                     random_order=random_sample, eye_threshold=eye_threshold)
    return jsonify({'total': total, 'images': rows})


@app.route('/api/labels.csv')
def api_labels():
    rows = library.export_labels(db())
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(rows[0].keys() if rows else ['path', 'label'])
    writer.writerows(tuple(r) for r in rows)
    response = send_file(io.BytesIO(out.getvalue().encode('utf-8')), mimetype='text/csv',
                         as_attachment=True, download_name='blur-labels.csv')
    response.headers['Cache-Control'] = 'no-store'
    return response


def checked_paths(paths):
    result = []
    for rel in paths:
        full = resolve(rel)
        if full is None or not os.path.isfile(full):
            abort(400, f'Not a file inside the images folder: {rel}')
        result.append(library.to_rel(BASE_DIR, full))
    return result


@app.route('/api/drafts', methods=['POST'])
def api_drafts():
    """Save review decisions as they are made, so a review can be continued later."""
    decisions = request.get_json(force=True).get('decisions', {})
    if not isinstance(decisions, dict) or any(d not in library.DECISIONS for d in decisions.values()):
        abort(400, f'decisions must be an object with values {library.DECISIONS}')
    paths = checked_paths(list(decisions))
    conn = db()
    library.save_drafts(conn, dict(zip(paths, decisions.values())))
    return jsonify({'saved': len(paths), 'drafts': library.draft_count(conn)})


@app.route('/api/apply', methods=['POST'])
def api_apply():
    if current_job is not None:
        abort(409, 'Wait until the scan has finished')
    data = request.get_json(force=True)
    move, keep = checked_paths(data.get('move', [])), checked_paths(data.get('keep', []))
    keep_blurry = checked_paths(data.get('keep_blurry', []))
    mode = 'random' if data.get('mode') == 'random' else 'threshold'
    # SQLite connections can't cross threads, so the worker thread opens its own
    batch, moved, errors = tpool.execute(
        lambda: library.apply_review(db(), BASE_DIR, move, keep, mode, keep_blurry))
    return jsonify({'batch': batch, 'moved': moved, 'kept': len(keep), 'kept_blurry': len(keep_blurry),
                    'errors': errors})


@app.route('/api/undo', methods=['POST'])
def api_undo():
    if current_job is not None:
        abort(409, 'Wait until the scan has finished')
    last = library.last_batch(db())
    if last is None:
        abort(400, 'Nothing to undo')
    restored, errors = tpool.execute(lambda: library.undo(db(), BASE_DIR, last['batch']))
    return jsonify({'restored': restored, 'errors': errors})


def render_preview(path, size):
    with Image.open(path) as img:
        img.draft('RGB', (size, size))  # fast reduced JPEG decoding
        img = ImageOps.exif_transpose(img).convert('RGB')
        img.thumbnail((size, size))
        buf = io.BytesIO()
        img.save(buf, 'JPEG', quality=85)
    return buf.getvalue()


@app.route('/thumb')
def thumb():
    path = resolve(request.args.get('path'))
    if path is None or not os.path.isfile(path):
        abort(404)
    try:
        size = min(max(int(request.args.get('size', 360)), 64), 2048)
    except ValueError:
        abort(400)
    try:
        data = tpool.execute(render_preview, path, size)
    except Exception:  # noqa: BLE001 -- unreadable file
        abort(415)
    response = send_file(io.BytesIO(data), mimetype='image/jpeg')
    response.headers['Cache-Control'] = 'private, max-age=86400'
    return response


# Formats every browser shows natively (including EXIF orientation); others are converted to JPEG
BROWSER_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def render_full(path):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert('RGB')
        buf = io.BytesIO()
        img.save(buf, 'JPEG', quality=92)
    return buf.getvalue()


@app.route('/original')
def original():
    """The photo in full resolution, for checking focus at 100 %."""
    path = resolve(request.args.get('path'))
    if path is None or not os.path.isfile(path) or not scoring.is_image(path):
        abort(404)
    if os.path.splitext(path)[1].lower() in BROWSER_FORMATS:
        return send_file(path, conditional=True, max_age=86400)
    try:
        data = tpool.execute(render_full, path)
    except Exception:  # noqa: BLE001 -- unreadable file
        abort(415)
    response = send_file(io.BytesIO(data), mimetype='image/jpeg')
    response.headers['Cache-Control'] = 'private, max-age=86400'
    return response


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
