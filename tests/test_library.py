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
    library.save_drafts(conn, {'blurry.jpg': False, 'trip/IMG_0001.JPG': True})
    library.save_drafts(conn, {'blurry.jpg': True})  # changed my mind
    assert library.draft_count(conn) == 2

    _, rows = library.candidates(conn, '', scoring.DEFAULT_THRESHOLD)
    assert {r['path']: r['draft'] for r in rows} == {'blurry.jpg': 1, 'trip/IMG_0001.JPG': 1}
    _, rows = library.candidates(conn, '', None, random_order=True, limit=2)
    assert {r['path'] for r in rows} == {'blurry.jpg', 'trip/IMG_0001.JPG'}  # drafts come first

    library.apply_review(conn, root, move=['blurry.jpg'], keep=['trip/IMG_0001.JPG'])
    assert library.draft_count(conn) == 0
