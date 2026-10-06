import os
import sys

import numpy as np
import pytest
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import library  # noqa: E402
import scoring  # noqa: E402


def make_photo(path, blur=0, seed=0):
    rng = np.random.default_rng(seed)
    img = Image.fromarray((rng.random((300, 400, 3)) * 255).astype(np.uint8)).resize((1600, 1200), Image.NEAREST)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, quality=90)


@pytest.fixture
def lib(tmp_path):
    root = tmp_path / 'images'
    make_photo(str(root / 'sharp.jpg'))
    make_photo(str(root / 'blurry.jpg'), blur=12)
    make_photo(str(root / 'trip' / 'IMG_0001.JPG'), blur=12)
    (root / 'trip' / 'IMG_0001.MOV').write_bytes(b'live photo video')
    (root / 'trip' / 'IMG_0001.AAE').write_bytes(b'edit sidecar')
    make_photo(str(root / 'trip' / 'IMG_0002.jpg'))
    make_photo(str(root / '@eaDir' / 'thumb.jpg'), blur=12)   # NAS thumbnail folder, must be skipped
    (root / 'notes.txt').write_text('not a photo')
    conn = library.connect(str(tmp_path / 'data' / 'db.sqlite'))
    return conn, str(root)


def test_score_separates_sharp_and_blurry(lib):
    _, root = lib
    sharp = scoring.score_file(os.path.join(root, 'sharp.jpg'))
    blurry = scoring.score_file(os.path.join(root, 'blurry.jpg'))
    assert sharp['error'] is None and blurry['error'] is None
    assert blurry['score'] < scoring.DEFAULT_THRESHOLD < sharp['score']


def test_unreadable_file_reports_error(tmp_path):
    bad = tmp_path / 'broken.jpg'
    bad.write_bytes(b'not really a jpeg')
    assert scoring.score_file(str(bad))['error']


def test_scan_is_incremental_and_skips_special_folders(lib):
    conn, root = lib
    assert library.scan(conn, root, log=lambda m: None) == 4
    paths = {r['path'] for r in conn.execute('SELECT path FROM images')}
    assert paths == {'sharp.jpg', 'blurry.jpg', 'trip/IMG_0001.JPG', 'trip/IMG_0002.jpg'}
    assert library.scan(conn, root, log=lambda m: None) == 0

    os.remove(os.path.join(root, 'sharp.jpg'))
    make_photo(os.path.join(root, 'trip', 'IMG_0002.jpg'), blur=12, seed=1)
    os.utime(os.path.join(root, 'trip', 'IMG_0002.jpg'), (1, 1))
    assert library.scan(conn, root, log=lambda m: None) == 1
    assert 'sharp.jpg' not in {r['path'] for r in conn.execute('SELECT path FROM images')}


def test_candidates_by_folder_and_threshold(lib):
    conn, root = lib
    library.scan(conn, root, log=lambda m: None)
    total, rows = library.candidates(conn, '', scoring.DEFAULT_THRESHOLD)
    assert total == 2 and {r['path'] for r in rows} == {'blurry.jpg', 'trip/IMG_0001.JPG'}
    total, rows = library.candidates(conn, 'trip', scoring.DEFAULT_THRESHOLD)
    assert [r['path'] for r in rows] == ['trip/IMG_0001.JPG']
    assert library.candidates(conn, 'tri', scoring.DEFAULT_THRESHOLD)[0] == 0


def test_move_with_companions_and_undo(lib):
    conn, root = lib
    library.scan(conn, root, log=lambda m: None)
    # A file with the same name already in Blurry must not be overwritten
    os.makedirs(os.path.join(root, 'trip', 'Blurry'))
    with open(os.path.join(root, 'trip', 'Blurry', 'IMG_0001.MOV'), 'wb') as f:
        f.write(b'older video')

    batch, moved, errors = library.apply_review(conn, root, move=['trip/IMG_0001.JPG'], keep=['blurry.jpg'])
    assert (moved, errors) == (1, [])
    blurry_dir = os.path.join(root, 'trip', 'Blurry')
    assert sorted(os.listdir(blurry_dir)) == ['IMG_0001.MOV', 'IMG_0001_1.AAE', 'IMG_0001_1.JPG', 'IMG_0001_1.MOV']
    assert sorted(os.listdir(os.path.join(root, 'trip'))) == ['Blurry', 'IMG_0002.jpg']
    labels = dict(conn.execute('SELECT path, label FROM images').fetchall())
    assert labels['trip/Blurry/IMG_0001_1.JPG'] == 'blurry' and labels['blurry.jpg'] == 'sharp'
    assert library.candidates(conn, '', scoring.DEFAULT_THRESHOLD)[0] == 0
    # Moved photos are not picked up again by the next scan
    assert library.scan(conn, root, log=lambda m: None) == 0

    assert library.last_batch(conn)['batch'] == batch
    restored, errors = library.undo(conn, root, batch)
    assert (restored, errors) == (2, [])  # the moved photo and the kept one
    labels = dict(conn.execute('SELECT path, label FROM images').fetchall())
    assert labels['trip/IMG_0001.JPG'] is None and labels['blurry.jpg'] is None
    assert sorted(os.listdir(os.path.join(root, 'trip'))) == ['Blurry', 'IMG_0001.AAE', 'IMG_0001.JPG',
                                                                'IMG_0001.MOV', 'IMG_0002.jpg']
    assert os.listdir(blurry_dir) == ['IMG_0001.MOV']
    assert library.last_batch(conn) is None
    assert library.candidates(conn, 'trip', scoring.DEFAULT_THRESHOLD)[0] == 1


def test_undo_removes_empty_blurry_folder(lib):
    conn, root = lib
    library.scan(conn, root, log=lambda m: None)
    batch, _, _ = library.apply_review(conn, root, move=['blurry.jpg'])
    assert os.path.isdir(os.path.join(root, 'Blurry'))
    library.undo(conn, root, batch)
    assert not os.path.exists(os.path.join(root, 'Blurry'))


def test_recommend_thresholds_per_camera(lib):
    conn, _ = lib
    rows = [(f'p{i}.jpg', float(i * 10), 'iPhone 13', 'blurry' if i < 15 else 'sharp') for i in range(50)]
    rows += [(f'q{i}.jpg', float(i * 10), 'X-S10', 'sharp') for i in range(50)]
    conn.executemany('INSERT INTO images (path, score, camera, label) VALUES (?, ?, ?, ?)', rows)
    rec = library.recommend_thresholds(conn)
    assert set(rec) == {'iPhone 13'}  # X-S10 has no blurry reviews yet
    assert rec['iPhone 13']['threshold'] == 150  # first sharp photo

    total, _ = library.candidates(conn, '', 100, camera_thresholds={'iPhone 13': 150}, include_reviewed=True)
    assert total == 15 + 10  # iPhone below 150, X-S10 below the default 100


def test_random_sample_and_label_export(lib):
    conn, root = lib
    library.scan(conn, root, log=lambda m: None)
    total, rows = library.candidates(conn, '', None, random_order=True, limit=10)
    assert total == 4 and len(rows) == 4  # all scores, not just below the threshold

    library.apply_review(conn, root, move=['blurry.jpg'], keep=['sharp.jpg'], mode='random')
    exported = {r['path']: (r['label'], r['mode']) for r in library.export_labels(conn)}
    assert exported == {'Blurry/blurry.jpg': ('blurry', 'random'), 'sharp.jpg': ('sharp', 'random')}
    assert library.last_batch(conn)['moved'] == 1 and library.last_batch(conn)['kept'] == 1


def test_drafts_survive_until_applied(lib):
    conn, root = lib
    library.scan(conn, root, log=lambda m: None)
    library.save_drafts(conn, {'blurry.jpg': 'keep', 'trip/IMG_0001.JPG': 'move'})
    library.save_drafts(conn, {'blurry.jpg': 'keep_blurry'})  # changed my mind
    assert library.draft_count(conn) == 2
    with pytest.raises(ValueError):
        library.save_drafts(conn, {'blurry.jpg': True})

    _, rows = library.candidates(conn, '', scoring.DEFAULT_THRESHOLD)
    assert {r['path']: r['draft'] for r in rows} == {'blurry.jpg': 'keep_blurry', 'trip/IMG_0001.JPG': 'move'}
    _, rows = library.candidates(conn, '', None, random_order=True, limit=2)
    assert {r['path'] for r in rows} == {'blurry.jpg', 'trip/IMG_0001.JPG'}  # drafts come first

    library.apply_review(conn, root, move=['blurry.jpg'], keep=['trip/IMG_0001.JPG'])
    assert library.draft_count(conn) == 0


def test_keep_blurry_and_undo_restores_decisions(lib):
    conn, root = lib
    library.scan(conn, root, log=lambda m: None)
    batch, moved, _ = library.apply_review(conn, root, move=['trip/IMG_0001.JPG'], keep=['sharp.jpg'],
                                           keep_blurry=['blurry.jpg'])
    assert moved == 1 and os.path.exists(os.path.join(root, 'blurry.jpg'))  # kept blurry: not moved
    labels = dict(conn.execute('SELECT path, label FROM images').fetchall())
    assert labels['blurry.jpg'] == 'blurry' and labels['trip/Blurry/IMG_0001.JPG'] == 'blurry'
    s = library.stats(conn)
    assert (s['moved'], s['kept'], s['kept_blurry']) == (1, 1, 1)
    last = library.last_batch(conn)
    assert (last['moved'], last['kept'], last['kept_blurry']) == (1, 1, 1)
    exported = {r['path']: (r['label'], r['moved']) for r in library.export_labels(conn)}
    assert exported == {'trip/Blurry/IMG_0001.JPG': ('blurry', 1), 'sharp.jpg': ('sharp', 0),
                        'blurry.jpg': ('blurry', 0)}

    # Undo brings the photos back with the decisions that were applied, ready to change and apply again
    assert library.undo(conn, root, batch) == (3, [])
    assert dict(conn.execute('SELECT path, decision FROM drafts').fetchall()) == {
        'trip/IMG_0001.JPG': 'move', 'sharp.jpg': 'keep', 'blurry.jpg': 'keep_blurry'}
    assert all(label is None for _, label in conn.execute('SELECT path, label FROM images'))


def test_eye_score_decides_when_there_is_a_face(lib):
    conn, _ = lib
    conn.executemany('INSERT INTO images (path, score, eye_score) VALUES (?, ?, ?)', [
        ('soft background.jpg', 50, 500),     # sharp eyes, rest out of focus: fine
        ('missed focus.jpg', 1000, 30),       # sharp background, blurry eyes: blurry
        ('landscape.jpg', 100, None),         # no face: judged on the score
        ('sharp landscape.jpg', 900, None),
    ])
    total, rows = library.candidates(conn, '', 200, eye_threshold=120)
    assert total == 2
    assert [r['path'] for r in rows] == ['missed focus.jpg', 'landscape.jpg']  # 30/120 before 100/200


def test_rescoring_keeps_review_decisions(lib):
    conn, root = lib
    library.scan(conn, root, log=lambda m: None)
    library.apply_review(conn, root, move=['blurry.jpg'], keep=['sharp.jpg'])
    conn.execute('UPDATE images SET score_version = 1, eye_score = -1')
    conn.commit()
    assert library.scan(conn, root, log=lambda m: None) == 4  # the moved photo in Blurry too
    rows = {r['path']: r for r in conn.execute('SELECT * FROM images')}
    assert rows['Blurry/blurry.jpg']['label'] == 'blurry' and rows['sharp.jpg']['label'] == 'sharp'
    assert all(r['score_version'] == scoring.SCORE_VERSION and r['eye_score'] is None for r in rows.values())
    assert library.scan(conn, root, log=lambda m: None) == 0

    make_photo(os.path.join(root, 'sharp.jpg'), blur=12, seed=1)  # edited: the old decision no longer applies
    os.utime(os.path.join(root, 'sharp.jpg'), (1, 1))
    library.scan(conn, root, log=lambda m: None)
    assert library.current_label(conn, 'sharp.jpg') is None


def test_old_database_is_upgraded(tmp_path):
    import sqlite3
    path = str(tmp_path / 'old.sqlite')
    old = sqlite3.connect(path)
    old.executescript('CREATE TABLE images (path TEXT PRIMARY KEY, mtime REAL, size INTEGER, score REAL, '
                      'global_score REAL, width INTEGER, height INTEGER, camera TEXT, taken TEXT, exposure REAL, '
                      'fnumber REAL, iso REAL, focal REAL, error TEXT, label TEXT, scored_at TEXT);'
                      'CREATE TABLE drafts (path TEXT PRIMARY KEY, move INTEGER, updated_at TEXT);'
                      "INSERT INTO drafts VALUES ('a.jpg', 1, ''), ('b.jpg', 0, '');")
    old.commit()
    old.close()
    conn = library.connect(path)
    assert dict(conn.execute('SELECT path, decision FROM drafts').fetchall()) == {'a.jpg': 'move', 'b.jpg': 'keep'}
    assert {'eye_score', 'faces', 'score_version'} <= {r['name'] for r in conn.execute('PRAGMA table_info(images)')}


def fake_faces(*faces):
    """A stand-in for the face detector: faces given as (x, y, width) fractions of the image."""
    def find(img):
        w, h = img.size
        return [np.array([x * w, y * h, fw * w, fw * w, x * w + 0.3 * fw * w, y * h + 0.4 * fw * w,
                          x * w + 0.7 * fw * w, y * h + 0.4 * fw * w, 0, 0, 0, 0, 0, 0, 0.95])
                for x, y, fw in faces]
    return find


def test_eye_score_measures_the_eyes(tmp_path, monkeypatch):
    # An upright 2400x3200 photo that is only sharp around one eye, stored sideways with an EXIF
    # orientation as cameras do; the eye crop must come from the right place. The face is 240 px wide,
    # so the crop needs more resolution than the reduced-size decoding used for the score.
    rng = np.random.default_rng(0)
    upright = np.full((3200, 2400), 128, np.uint8)
    upright[1020:1095, 1090:1170] = (rng.random((75, 80)) * 255).astype(np.uint8)
    exif = Image.Exif()
    exif[scoring.TAG_ORIENTATION] = 6
    path = str(tmp_path / 'portrait.jpg')
    Image.fromarray(upright).transpose(Image.Transpose.ROTATE_90).save(path, quality=95, exif=exif)

    # Eyes at x = 0.43 and 0.47 of the width, y = 0.3 of the height + 0.04 of the width
    monkeypatch.setattr(scoring, 'find_faces', fake_faces((0.4, 0.3, 0.1)))
    result = scoring.score_file(path)
    assert result['faces'] == 1 and result['eye_score'] > 1000
    assert abs(result['eye_x'] - 0.47) < 0.005 and abs(result['eye_y'] - 0.33) < 0.005  # the sharp one

    monkeypatch.setattr(scoring, 'find_faces', fake_faces((0.4, 0.6, 0.1)))  # eyes on the plain part
    assert scoring.score_file(path)['eye_score'] < 1

    monkeypatch.setattr(scoring, 'find_faces', fake_faces((0.5, 0.5, 0.04)))  # a face in the background
    result = scoring.score_file(path)
    assert result['eye_score'] is None and result['faces'] == 0


def test_photos_without_faces_have_no_eye_score(lib):
    _, root = lib
    result = scoring.score_file(os.path.join(root, 'sharp.jpg'))
    assert result['eye_score'] is None and result['faces'] == 0
