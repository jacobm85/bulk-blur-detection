"""Sharpness scoring for a single image file.

The score is the 90th percentile of the Laplacian variance over a 6x6 grid of tiles, computed on a
copy of the image downscaled to at most 1024 px. Lower = blurrier. Using tiles means a photo with a
sharp subject and a soft background still scores as sharp, and downscaling makes the score
comparable between cameras with different resolutions.
"""
import os

import numpy as np
from PIL import Image

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:  # HEIC support is optional
    pass

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.tif', '.tiff', '.webp', '.bmp'}
ANALYSIS_SIZE = 1024
GRID = 6
MIN_SIZE = 64

# Calibrated on the CERTH NaturalBlurSet (tools/evaluate_certh.py)
DEFAULT_THRESHOLD = 200

EXIF_IFD = 0x8769
TAG_MAKE, TAG_MODEL = 271, 272
TAG_DATETIME, TAG_DATETIME_ORIGINAL = 306, 36867
TAG_EXPOSURE, TAG_FNUMBER, TAG_ISO, TAG_FOCAL = 33434, 33437, 34855, 37386


def is_image(name):
    return os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS


def _clean(value):
    return str(value).replace('\x00', '').strip() if value is not None else ''


def _number(value):
    try:
        if isinstance(value, (tuple, list)):
            value = value[0]
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def read_exif(img):
    exif = img.getexif()
    sub = exif.get_ifd(EXIF_IFD)
    make, model = _clean(exif.get(TAG_MAKE)), _clean(exif.get(TAG_MODEL))
    if make and model and make.split()[0].lower() not in model.lower():
        camera = f'{make} {model}'
    else:
        camera = model or make or None
    return {
        'camera': camera,
        'taken': _clean(sub.get(TAG_DATETIME_ORIGINAL) or exif.get(TAG_DATETIME)) or None,
        'exposure': _number(sub.get(TAG_EXPOSURE)),
        'fnumber': _number(sub.get(TAG_FNUMBER)),
        'iso': _number(sub.get(TAG_ISO)),
        'focal': _number(sub.get(TAG_FOCAL)),
    }


def load_gray(img, max_size=ANALYSIS_SIZE):
    """Grayscale copy of img with the longest side at most max_size, as float64."""
    img.draft('L', (max_size, max_size))  # fast reduced JPEG decoding; no-op for other formats
    gray = img.convert('L')
    gray.thumbnail((max_size, max_size), Image.BOX)
    return np.asarray(gray, dtype=np.float64)


def laplacian(gray):
    p = np.pad(gray, 1, mode='reflect')
    return p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * gray


def sharpness(gray, grid=GRID):
    """Return (tile score, global Laplacian variance) for a grayscale image."""
    lap = laplacian(gray)
    h, w = lap.shape
    tiles = [lap[i * h // grid:(i + 1) * h // grid, j * w // grid:(j + 1) * w // grid].var()
             for i in range(grid) for j in range(grid)]
    return float(np.percentile(tiles, 90)), float(lap.var())


def score_file(path):
    """Score one image file. Returns a dict; on failure it only contains 'error'."""
    try:
        with Image.open(path) as img:
            width, height = img.size
            info = read_exif(img)
            if min(width, height) < MIN_SIZE:
                return {'error': 'image too small', 'width': width, 'height': height, **info}
            score, global_score = sharpness(load_gray(img))
        return {'score': score, 'global_score': global_score, 'width': width, 'height': height,
                'error': None, **info}
    except Exception as e:  # noqa: BLE001 -- one bad file must not stop a scan of thousands
        return {'error': f'{type(e).__name__}: {e}'}
