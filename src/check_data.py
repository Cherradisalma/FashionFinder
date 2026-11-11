import os
from PIL import Image

DATA_DIR = "data/raw"
for classe in sorted(os.listdir(DATA_DIR)):
    path = os.path.join(DATA_DIR, classe)
    if not os.path.isdir(path):
        continue
    files = os.listdir(path)
    bad = []
    for f in files:
        try:
            Image.open(os.path.join(path, f)).verify()
        except Exception:
            bad.append(f)
    print(f"{classe:12s} {len(files):4d} images, corrompues : {bad}")