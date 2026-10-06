# Bulk Blur Detector Web GUI

Find the blurry photos in a large photo library and move them out of the way, from a web page in Docker.

Having around 200k unsorted photos, with monthly new additions, is a pain to handle. This tool scores
every photo for sharpness, shows you the blurriest ones and moves the ones you confirm to a `Blurry`
folder next to them. Nothing is moved without your review, and every move can be undone.

Originally based on https://github.com/danngalann/bulk-blur-detection and
https://github.com/Utkarsh-Deshmukh/Blurry-Image-Detector. Version 1 (Laplacian threshold plus the DCT
model, moving photos automatically) is kept on the [`v1` branch](../../tree/v1).

## Usage
1. **Folder** – browse to the folder you want to clean up.
2. **Scan** – scores every photo in the folder and its subfolders. Scores are stored in a database, so
   later scans only look at new or changed photos. Nothing is moved.
3. **Review** – the photos below the threshold are shown blurriest first. All are marked *Move*; click
   the ones you want to *Keep* and press *Apply*. Shift-click (or B in the large view) marks a photo
   *Keep blurry*: it stays where it is, but is recorded as blurry, so photos you keep for what they show
   don't end up as "sharp" in the training data. Click the magnifier to see a photo larger.
   - Moved photos go to a `Blurry` folder in the same folder as the photo. Files that belong to the photo
     move with it: iPhone Live Photo videos (`.MOV`), edit sidecars (`.AAE`), RAW files from RAW+JPEG
     shooting and `.xmp` files. Existing files are never overwritten.
   - Every Move/Keep click is saved right away, so you can stop in the middle of a review and continue
     later; only *Apply* moves files.
   - *Undo last apply* puts the last batch back, including the photos you marked Keep, and restores your
     Move/Keep decisions for them so you can correct a few and apply again.
   - Photos you keep are remembered and not shown again.
   - The magnifier opens a large view. Click the photo to see it at 100 % (one photo pixel per screen
     pixel) centred on where you clicked, drag to pan, click again to fit. Z opens 100 % at the sharpest
     eye when the photo has a face. Previous/next stays at 100 % on the same spot (or the next photo's
     eye). HEIC and TIFF are converted on the fly. Keys: ← → previous/next, M move, K keep, B keep blurry,
     Z 100 %/fit, Esc close.

The score finds the candidates and you verify them: the same two-step idea as version 1, where the DCT
model double-checked the Laplacian result, but with a person as the second step. Measured on the CERTH
dataset, the DCT model removed only 3 of 46 wrongly flagged photos while taking about 2 s per photo.

### Collecting training data
Every Move/Keep decision is stored and can be downloaded as *labels (CSV)* from the Scan section, to
train a better model later. Reviewing only photos below the threshold never shows blurry photos the
score missed, so also use the review mode **Random sample (training data)** now and then: it shows
random photos of any score, all marked Keep, and you mark the blurry ones. The CSV records which mode
each decision came from.

Supported formats: JPEG, HEIC/HEIF (iPhone), PNG, TIFF, WebP, BMP. NAS system folders such as `@eaDir`
and `#recycle` are skipped.

### Threshold and per-camera suggestions
The score is the sharpness of the sharpest parts of the photo (Laplacian variance over a grid of tiles,
on a copy downscaled to 1024 px). Lower = blurrier. The default threshold of 200 was calibrated on the
CERTH Image Blur Dataset, where it finds about two thirds of the blurry photos while flagging about 5 %
of the sharp ones (version 1 found 58 % while moving 8 % of the sharp ones).

Cameras differ, so once you have reviewed about 40 photos from a camera (at least 10 sharp and 10
blurry) the review page suggests a threshold for that camera. Tick *Use suggested thresholds* to use them.

### Faces: the eyes decide
In a portrait the background is often meant to be soft, and a portrait with sharp hair and a soft eye
is a miss. So photos where a face is a clear part of the picture (at least 6 % of the photo's width) are
judged on the eyes instead: faces and eyes are found with the [YuNet](models/README.md) face detector,
and the *eye score* is the sharpness around the sharper eye, measured in the full-resolution photo at a
fixed face size, so it is comparable between close-ups and half-body shots. Photos with an eye score
are compared with the *eye threshold* (default 150), all others with the normal threshold. People in
the background of a street scene, faces turned away and very small faces don't count, and those photos
are judged on the whole photo as before. The eye threshold also gets a suggestion from your reviews,
after about 40 reviewed photos with a face.

The review page shows both scores (`eyes 85 · whole 340`). A new version of the scoring scores every
photo again on the next scan; your review decisions are kept.

## Docker
Use git as source in Portainer, or `docker compose up -d` with `docker-compose.yml`.
Specify the path where your photos are stored in the compose file. The database with scores and review
decisions is kept in the `blur-data` volume.
```
Exposes port 5050
```

## Development
```
pip install -r requirements.txt pytest
python -m pytest tests
IMAGES_DIR=/path/to/photos python app.py
```
`tools/evaluate_certh.py` measures the score against the CERTH Image Blur Dataset; run it after any
change to `scoring.py`. Raise `SCORE_VERSION` in `scoring.py` when a change should make the next scan
score every photo again. Photos of your own for trying things out can go in `testbilder/`, which git
ignores.

## Requirements
 ```Docker```
