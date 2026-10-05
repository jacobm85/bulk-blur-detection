"""Photo library database: incremental scanning, review decisions, moving and undo.

All paths stored in the database are relative to the images root and use '/' separators, so the
database stays valid if the library is mounted somewhere else.

Command line (used by the web app, also usable on its own):
    python library.py scan --root /app/images --folder 2024 --db /app/data/blurdetector.db
"""
import argparse
import os
import shutil
import sqlite3
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

from scoring import is_image, score_file

BLURRY_FOLDER_NAME = 'Blurry'
# Files with the same name as a photo that belong to it and must move with it:
# Live Photo videos, iPhone edit sidecars, RAW files from RAW+JPEG shooting, XMP metadata.
COMPANION_EXTENSIONS = {'.mov', '.aae', '.xmp', '.dng', '.raf', '.cr2', '.cr3', '.nef', '.arw', '.orf', '.rw2'}

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    path TEXT PRIMARY KEY,
    mtime REAL, size INTEGER,
    score REAL, global_score REAL,
    width INTEGER, height INTEGER,
    camera TEXT, taken TEXT, exposure REAL, fnumber REAL, iso REAL, focal REAL,
    error TEXT,
    label TEXT,          -- NULL = not reviewed, 'sharp' = kept after review, 'blurry' = moved
    scored_at TEXT
);
-- Every review decision, so an apply can be undone completely and training data can account for
-- how photos were picked for review ('threshold' = below the threshold, 'random' = random sample)
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY,
    batch TEXT, image TEXT, label TEXT, prev_label TEXT, mode TEXT,
    reviewed_at TEXT, undone_at TEXT
);
CREATE INDEX IF NOT EXISTS reviews_batch ON reviews(batch);
CREATE TABLE IF NOT EXISTS moves (
    id INTEGER PRIMARY KEY,
    batch TEXT, src TEXT, dst TEXT, image TEXT,  -- image = path of the photo this file belongs to
    moved_at TEXT, undone_at TEXT
);
CREATE INDEX IF NOT EXISTS moves_batch ON moves(batch);
"""
SCORE_FIELDS = ['score', 'global_score', 'width', 'height', 'camera', 'taken',
                'exposure', 'fnumber', 'iso', 'focal', 'error']


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def connect(db_path):
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    conn.executescript(SCHEMA)
    return conn


def to_rel(root, path):
    rel = os.path.relpath(path, root).replace(os.sep, '/')
    return '' if rel == '.' else rel


def to_abs(root, rel):
    return os.path.join(root, *rel.split('/')) if rel else root


def in_folder_sql(folder):
    """SQL condition + params matching paths inside folder (recursively)."""
    if not folder:
        return '1=1', []
    return 'substr(path, 1, ?) = ?', [len(folder) + 1, folder + '/']


def skip_dir(name):
    # Blurry folders hold photos already moved; '@eaDir' and '#recycle' are NAS thumbnail/trash folders
    return name == BLURRY_FOLDER_NAME or name.startswith(('.', '@', '#'))


def walk_images(root, folder):
    for dirpath, dirnames, filenames in os.walk(to_abs(root, folder)):
        dirnames[:] = sorted(d for d in dirnames if not skip_dir(d))
        for name in sorted(filenames):
            if is_image(name) and not name.startswith('.'):
                yield os.path.join(dirpath, name)


def scan(conn, root, folder='', workers=None, log=print):
    """Score new and changed images under folder; forget images that no longer exist."""
    log(f'[scan] Looking for images in /{folder}')
    known = {r['path']: (r['mtime'], r['size'])
             for r in conn.execute(f'SELECT path, mtime, size FROM images WHERE {in_folder_sql(folder)[0]}',
                                   in_folder_sql(folder)[1])}
    seen, todo = set(), []
    for path in walk_images(root, folder):
        rel = to_rel(root, path)
        try:
            st = os.stat(path)
        except OSError:  # removed while scanning
            continue
        seen.add(rel)
        if known.get(rel) != (st.st_mtime, st.st_size):
            todo.append((rel, path, st.st_mtime, st.st_size))
    log(f'[scan] {len(seen)} images found, {len(todo)} new or changed')

    # Rows outside Blurry folders whose file is gone were deleted or moved by hand
    gone = [p for p in known if p not in seen and f'/{BLURRY_FOLDER_NAME}/' not in f'/{p}'
            and not os.path.exists(to_abs(root, p))]
    conn.executemany('DELETE FROM images WHERE path = ?', [(p,) for p in gone])
    conn.commit()
    if gone:
        log(f'[scan] {len(gone)} images no longer exist and were removed from the database')

    start, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for (rel, path, mtime, size), result in zip(todo, pool.map(score_file, [t[1] for t in todo], chunksize=16)):
            values = [result.get(f) for f in SCORE_FIELDS]
            conn.execute(
                f'INSERT INTO images (path, mtime, size, {", ".join(SCORE_FIELDS)}, label, scored_at) '
                f'VALUES (?, ?, ?, {", ".join("?" * len(SCORE_FIELDS))}, NULL, ?) '
                f'ON CONFLICT(path) DO UPDATE SET mtime=excluded.mtime, size=excluded.size, '
                f'{", ".join(f"{f}=excluded.{f}" for f in SCORE_FIELDS)}, label=NULL, scored_at=excluded.scored_at',
                [rel, mtime, size, *values, now()])
            done += 1
            if done % 100 == 0 or done == len(todo):
                conn.commit()
                rate = done / max(time.time() - start, 1e-6)
                log(f'[scan] {done}/{len(todo)} scored ({rate:.1f} images/s, '
                    f'~{(len(todo) - done) / rate / 60:.0f} min left)')
    conn.commit()
    errors = conn.execute(f'SELECT COUNT(*) FROM images WHERE error IS NOT NULL AND {in_folder_sql(folder)[0]}',
                          in_folder_sql(folder)[1]).fetchone()[0]
    log(f'[scan] Done. {errors} images could not be read.' if errors else '[scan] Done.')
    return len(todo)


def candidates(conn, folder='', threshold=None, camera=None, include_reviewed=False, limit=60, offset=0,
               camera_thresholds=None, random_order=False):
    """Unreviewed images below threshold in folder, blurriest first, plus the total count.
    camera_thresholds ({camera: threshold}) overrides threshold for those cameras.
    random_order returns a random sample instead (use threshold=None to sample all scores)."""
    where, params = in_folder_sql(folder)
    where += f" AND error IS NULL AND '/' || path NOT LIKE '%/{BLURRY_FOLDER_NAME}/%'"
    if threshold is not None and camera_thresholds:
        cases = ' '.join('WHEN ? THEN ?' for _ in camera_thresholds)
        where += f" AND score < CASE COALESCE(camera, '(unknown)') {cases} ELSE ? END"
        for cam, t in camera_thresholds.items():
            params += [cam, t]
        params.append(threshold)
    elif threshold is not None:
        where += ' AND score < ?'
        params.append(threshold)
    if camera:
        where += ' AND camera IS ?' if camera != '(unknown)' else ' AND camera IS NULL'
        if camera != '(unknown)':
            params.append(camera)
    if not include_reviewed:
        where += ' AND label IS NULL'
    total = conn.execute(f'SELECT COUNT(*) FROM images WHERE {where}', params).fetchone()[0]
    order = 'random()' if random_order else 'score'
    rows = conn.execute(f'SELECT * FROM images WHERE {where} ORDER BY {order} LIMIT ? OFFSET ?',
                        params + [limit, 0 if random_order else offset]).fetchall()
    return total, [dict(r) for r in rows]


def recommend_thresholds(conn, window=20, min_reviewed=40, min_each=10):
    """Per-camera threshold suggestion learned from review decisions.

    Reviewed photos of a camera are sorted by score; the suggestion is the highest score that is
    the midpoint of `window` consecutive reviewed photos of which at least half were marked blurry,
    i.e. roughly where moved photos stop outnumbering kept ones. Above that point,
    most photos you looked at were sharp, so reviewing them is mostly wasted effort. Needs at least
    `min_reviewed` reviewed photos with `min_each` of each label. Returns {camera: {...}}.
    """
    result = {}
    rows = conn.execute("SELECT COALESCE(camera, '(unknown)') AS camera, score, label FROM images "
                        "WHERE label IS NOT NULL AND error IS NULL ORDER BY camera, score").fetchall()
    by_camera = {}
    for r in rows:
        by_camera.setdefault(r['camera'], []).append((r['score'], r['label'] == 'blurry'))
    for camera, items in by_camera.items():
        blurry = sum(b for _, b in items)
        if len(items) < min_reviewed or blurry < min_each or len(items) - blurry < min_each:
            continue
        best = None
        for i in range(window, len(items) + 1):
            if sum(b for _, b in items[i - window:i]) * 2 >= window:
                best = items[i - window // 2][0]
        if best is not None:
            result[camera] = {'threshold': round(best), 'reviewed': len(items), 'blurry': blurry}
    return result


def stats(conn, folder=''):
    where, params = in_folder_sql(folder)
    row = conn.execute(
        f"SELECT COUNT(*) AS images, SUM(error IS NOT NULL) AS errors, SUM(label = 'sharp') AS kept, "
        f"SUM(label = 'blurry') AS moved FROM images WHERE {where}", params).fetchone()
    cameras = conn.execute(
        f"SELECT COALESCE(camera, '(unknown)') AS camera, COUNT(*) AS images FROM images "
        f"WHERE {where} AND error IS NULL GROUP BY 1 ORDER BY 2 DESC", params).fetchall()
    return {**{k: row[k] or 0 for k in row.keys()}, 'cameras': [dict(c) for c in cameras]}


def companions(path):
    """Other files next to path with the same name that belong to the same photo."""
    folder, name = os.path.split(path)
    stem = os.path.splitext(name)[0].lower()
    result = []
    for other in os.listdir(folder):
        o_stem, o_ext = os.path.splitext(other)
        if other != name and o_stem.lower() == stem and o_ext.lower() in COMPANION_EXTENSIONS:
            result.append(os.path.join(folder, other))
        elif other.lower() == name.lower() + '.xmp':  # darktable/digiKam style photo.jpg.xmp
            result.append(os.path.join(folder, other))
    return sorted(result)


def free_stem(dst_folder, stem, extensions):
    """A file stem that doesn't collide with existing files for any of the given extensions."""
    candidate, counter = stem, 1
    while any(os.path.exists(os.path.join(dst_folder, candidate + ext)) for ext in extensions):
        candidate = f'{stem}_{counter}'
        counter += 1
    return candidate


def move_group(files, dst_folder):
    """Move a photo and its companions into dst_folder, keeping their names paired.
    Returns a list of (src, dst) absolute paths."""
    os.makedirs(dst_folder, exist_ok=True)
    stem = os.path.splitext(os.path.basename(files[0]))[0]
    tails = [os.path.basename(f)[len(stem):] for f in files]  # e.g. '.jpg', '.MOV', '.jpg.xmp'
    new_stem = free_stem(dst_folder, stem, tails)
    moved = []
    for f, tail in zip(files, tails):
        dst = os.path.join(dst_folder, new_stem + tail)
        shutil.move(f, dst)
        moved.append((f, dst))
    return moved


def current_label(conn, rel):
    row = conn.execute('SELECT label FROM images WHERE path = ?', (rel,)).fetchone()
    return row['label'] if row else None


def apply_review(conn, root, move=(), keep=(), mode='threshold'):
    """Move the photos in `move` (with companions) to a Blurry folder next to them and label them
    blurry; label the photos in `keep` as sharp. `mode` records how the photos were picked for
    review. Returns (batch id, number moved, errors)."""
    batch, moved, errors = uuid.uuid4().hex[:12], 0, []
    for rel in move:
        src = to_abs(root, rel)
        prev = current_label(conn, rel)
        try:
            pairs = move_group([src] + companions(src), os.path.join(os.path.dirname(src), BLURRY_FOLDER_NAME))
        except OSError as e:
            errors.append(f'{rel}: {e}')
            continue
        new_rel = to_rel(root, pairs[0][1])
        conn.execute("UPDATE images SET path = ?, label = 'blurry' WHERE path = ?", (new_rel, rel))
        conn.executemany('INSERT INTO moves (batch, src, dst, image, moved_at) VALUES (?, ?, ?, ?, ?)',
                         [(batch, to_rel(root, s), to_rel(root, d), new_rel, now()) for s, d in pairs])
        conn.execute("INSERT INTO reviews (batch, image, label, prev_label, mode, reviewed_at) "
                     "VALUES (?, ?, 'blurry', ?, ?, ?)", (batch, new_rel, prev, mode, now()))
        conn.commit()
        moved += 1
    for rel in keep:
        prev = current_label(conn, rel)
        conn.execute("UPDATE images SET label = 'sharp' WHERE path = ?", (rel,))
        conn.execute("INSERT INTO reviews (batch, image, label, prev_label, mode, reviewed_at) "
                     "VALUES (?, ?, 'sharp', ?, ?, ?)", (batch, rel, prev, mode, now()))
    conn.commit()
    return batch, moved, errors


def last_batch(conn):
    row = conn.execute("SELECT batch, SUM(label = 'blurry') AS moved, SUM(label = 'sharp') AS kept, "
                       "MAX(reviewed_at) AS reviewed_at FROM reviews WHERE undone_at IS NULL "
                       "GROUP BY batch ORDER BY MAX(id) DESC LIMIT 1").fetchone()
    return dict(row) if row else None


def undo(conn, root, batch):
    """Undo an apply: move every file of the batch back to where it came from and reset the labels
    of the photos that were kept. Returns (restored photos, errors)."""
    errors, restored = [], set()
    for m in conn.execute('SELECT * FROM moves WHERE batch = ? AND undone_at IS NULL ORDER BY id', (batch,)).fetchall():
        src, dst = to_abs(root, m['src']), to_abs(root, m['dst'])
        if os.path.exists(src):
            errors.append(f"{m['src']} already exists, left {m['dst']} in place")
            continue
        try:
            os.makedirs(os.path.dirname(src), exist_ok=True)
            shutil.move(dst, src)
        except OSError as e:
            errors.append(f"{m['dst']}: {e}")
            continue
        conn.execute('UPDATE moves SET undone_at = ? WHERE id = ?', (now(), m['id']))
        if m['dst'] == m['image']:
            conn.execute('UPDATE images SET path = ? WHERE path = ?', (m['src'], m['image']))
        conn.commit()
        folder = os.path.dirname(dst)
        if os.path.basename(folder) == BLURRY_FOLDER_NAME and not os.listdir(folder):
            os.rmdir(folder)  # don't leave empty Blurry folders behind

    for r in conn.execute('SELECT * FROM reviews WHERE batch = ? AND undone_at IS NULL', (batch,)).fetchall():
        if r['label'] == 'sharp':
            conn.execute('UPDATE images SET label = ? WHERE path = ?', (r['prev_label'], r['image']))
        else:
            moved_back = conn.execute('SELECT src FROM moves WHERE batch = ? AND dst = ? AND undone_at IS NOT NULL',
                                      (batch, r['image'])).fetchone()
            if moved_back is None:
                continue  # the move back failed; leave it so undo can be retried
            conn.execute('UPDATE images SET label = ? WHERE path = ?', (r['prev_label'], moved_back['src']))
        restored.add(r['image'])
        conn.execute('UPDATE reviews SET undone_at = ? WHERE id = ?', (now(), r['id']))
    conn.commit()
    return len(restored), errors


def export_labels(conn):
    """All reviewed photos with their label, score and EXIF, for training a model."""
    return conn.execute(
        "SELECT i.path, i.label, i.score, i.global_score, i.camera, i.taken, i.exposure, i.fnumber, i.iso, "
        "i.focal, i.width, i.height, r.mode, r.reviewed_at FROM images i "
        "LEFT JOIN reviews r ON r.id = (SELECT MAX(id) FROM reviews WHERE image = i.path AND undone_at IS NULL) "
        "WHERE i.label IS NOT NULL ORDER BY r.reviewed_at").fetchall()


def main():
    parser = argparse.ArgumentParser(description='Blur detector photo library')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('scan', help='Score new and changed images')
    p.add_argument('--root', required=True, help='Images root directory')
    p.add_argument('--folder', default='', help='Folder inside root to scan (default: everything)')
    p.add_argument('--db', required=True, help='SQLite database file')
    p.add_argument('--workers', type=int, default=None, help='Parallel processes (default: all CPU cores)')
    args = parser.parse_args()

    conn = connect(args.db)
    scan(conn, os.path.realpath(args.root), args.folder.strip('/'), args.workers,
         log=lambda msg: print(msg, flush=True))


if __name__ == '__main__':
    sys.exit(main())
