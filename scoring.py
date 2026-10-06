"""Sharpness scoring for a single image file.

The score is the 90th percentile of the Laplacian variance over a 6x6 grid of tiles, computed on a
copy of the image downscaled to at most 1024 px. Lower = blurrier. Using tiles means a photo with a
sharp subject and a soft background still scores as sharp, and downscaling makes the score
comparable between cameras with different resolutions.

Photos where a face is a clear part of the picture also get an eye score: the Laplacian variance
around the sharper eye, measured on a crop from the full-resolution photo that is scaled to a fixed
face size. For those photos the eye score decides, so a portrait with a soft background is fine and
one where the focus landed behind the face is not. Faces are found with the YuNet detector from
OpenCV; without OpenCV the eye score is left out and the photo is judged on the score alone.
"""
import math
import os

import numpy as np
from PIL import Image

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:  # HEIC support is optional
    pass

try:
    import cv2
except ImportError:  # eye scores are optional
    cv2 = None

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.tif', '.tiff', '.webp', '.bmp'}
ANALYSIS_SIZE = 1024
GRID = 6
MIN_SIZE = 64

# Calibrated on the CERTH NaturalBlurSet (tools/evaluate_certh.py)
DEFAULT_THRESHOLD = 200
# From the 14 CERTH photos with a clear face (all blurry ones <= 136, sharp ones >= 181); only a first
# guess, the review page suggests a better one from your own decisions
DEFAULT_EYE_THRESHOLD = 150
# Raise when scoring changes, so the next scan scores every photo again (review decisions are kept)
SCORE_VERSION = 2

FACE_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'face_detection_yunet_2023mar.onnx')
FACE_MIN_CONFIDENCE = 0.8
FACE_SEARCH_SIZE = 640   # faces are searched in a copy this size; finds the same faces as 1024 px, 3x faster
# Only faces that are a clear part of the picture count: people in the background of a street scene
# don't tell whether the photo is in focus
FACE_MIN_SHARE = 0.06    # face width relative to the long side of the photo
FACE_SIZE = 200          # eye crops are measured as if the face were this many pixels wide
EYE_CROP = 0.24          # side of the square crop around an eye, relative to the face width

EXIF_IFD = 0x8769
TAG_MAKE, TAG_MODEL, TAG_ORIENTATION = 271, 272, 274
TAG_DATETIME, TAG_DATETIME_ORIGINAL = 306, 36867
TAG_EXPOSURE, TAG_FNUMBER, TAG_ISO, TAG_FOCAL = 33434, 33437, 34855, 37386

# EXIF orientation -> transpose that turns the stored pixels upright
ORIENTATIONS = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
                4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
                6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE, 8: Image.Transpose.ROTATE_90}


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


def load_rgb(img, min_size):
    """img decoded as RGB. JPEGs are decoded at a reduced size (1/2, 1/4 or 1/8) as long as both
    sides stay at least min_size (width, height); other formats are decoded in full."""
    img.draft('RGB', min_size)
    return img.convert('RGB')


def upright(img, orientation):
    return img.transpose(ORIENTATIONS[orientation]) if orientation in ORIENTATIONS else img


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


_detector = None


def opencv_version():
    return tuple(int(part) for part in cv2.__version__.split('.')[:2])


def face_detector():
    """The YuNet face detector, created once per process; None if OpenCV or the model is missing."""
    global _detector
    if _detector is None:
        _detector = False
        # The model needs OpenCV 4.8; older versions load it but fail on the first photo
        if cv2 is not None and os.path.exists(FACE_MODEL) and opencv_version() >= (4, 8):
            cv2.setNumThreads(1)  # scans already run one process per core
            _detector = cv2.FaceDetectorYN.create(FACE_MODEL, '', (320, 320), FACE_MIN_CONFIDENCE)
    return _detector or None


def find_faces(img):
    """Faces in an upright RGB image: rows of x, y, w, h, right eye x, y, left eye x, y, nose x, y,
    mouth corners x, y, x, y, confidence."""
    detector = face_detector()
    if detector is None:
        return []
    detector.setInputSize(img.size)
    _, faces = detector.detect(np.ascontiguousarray(np.asarray(img)[:, :, ::-1]))  # OpenCV wants BGR
    return [] if faces is None else list(faces)


def eye_sharpness(photo, eye, face_width):
    """Laplacian variance of the square around one eye, measured as if the face were FACE_SIZE px
    wide, or None if the square doesn't fit inside the photo."""
    half = EYE_CROP * face_width / 2
    box = (round(eye[0] - half), round(eye[1] - half), round(eye[0] + half), round(eye[1] + half))
    if box[0] < 0 or box[1] < 0 or box[2] > photo.width or box[3] > photo.height:
        return None
    side = round(EYE_CROP * FACE_SIZE)
    crop = photo.crop(box).convert('L').resize((side, side), Image.BOX)
    return float(laplacian(np.asarray(crop, dtype=np.float64)).var())


def eye_score(path, decoded, search, orientation, full_size):
    """Sharpness of the sharpest eye among the faces that are a clear part of the photo.

    decoded is the photo as decoded for analysis (possibly at a reduced size, not upright), search
    the upright downscaled copy to look for faces in and full_size the upright size of the photo.
    Returns (score, x, y, faces) with the position of that eye as fractions of the upright photo,
    or (None, None, None, 0) when no face counts."""
    to_full = full_size[0] / search.width
    min_width = max(FACE_SIZE, FACE_MIN_SHARE * max(full_size))
    faces = [f for f in find_faces(search) if f[2] * to_full >= min_width]
    if not faces:
        return None, None, None, 0

    # The eye crops need a resolution where the smallest face is at least FACE_SIZE px wide
    needed = FACE_SIZE / min(f[2] * to_full for f in faces)
    photo = decoded
    if max(photo.size) < needed * max(full_size) - 1:
        with Image.open(path) as img:
            photo = load_rgb(img, (math.ceil(needed * img.width), math.ceil(needed * img.height)))
    photo = upright(photo, orientation)
    to_photo = photo.width / search.width

    best = (None, None, None)
    for f in faces:
        for eye in (f[4:6], f[6:8]):
            score = eye_sharpness(photo, eye * to_photo, f[2] * to_photo)
            if score is not None and (best[0] is None or score > best[0]):
                best = (score, float(eye[0]) / search.width, float(eye[1]) / search.height)
    if best[0] is None:  # all eyes at the edge of the photo
        return None, None, None, 0
    return (*best, len(faces))


def score_file(path):
    """Score one image file. Returns a dict; on failure it only contains 'error'."""
    try:
        with Image.open(path) as img:
            width, height = img.size
            info = read_exif(img)
            if min(width, height) < MIN_SIZE:
                return {'error': 'image too small', 'width': width, 'height': height, **info}
            orientation = img.getexif().get(TAG_ORIENTATION, 1)
            decoded = load_rgb(img, (ANALYSIS_SIZE, ANALYSIS_SIZE))
            analysis = decoded.copy()
            analysis.thumbnail((ANALYSIS_SIZE, ANALYSIS_SIZE), Image.BOX)
            score, global_score = sharpness(np.asarray(analysis.convert('L'), dtype=np.float64))
            search = upright(analysis, orientation)
            search.thumbnail((FACE_SEARCH_SIZE, FACE_SEARCH_SIZE), Image.BOX)
            full_size = (height, width) if orientation in (5, 6, 7, 8) else (width, height)
            eye, eye_x, eye_y, faces = eye_score(path, decoded, search, orientation, full_size)
        return {'score': score, 'global_score': global_score, 'eye_score': eye, 'eye_x': eye_x, 'eye_y': eye_y,
                'faces': faces, 'width': width, 'height': height, 'error': None, **info}
    except Exception as e:  # noqa: BLE001 -- one bad file must not stop a scan of thousands
        return {'error': f'{type(e).__name__}: {e}'}
