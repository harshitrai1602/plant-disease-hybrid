from pathlib import Path
from PIL import Image
IMG_EXTS={".jpg",".jpeg",".png",".bmp",".JPG",".JPEG",".PNG"}
for root in ["data/raw/plantvillage","data/raw/plantdoc"]:
    p=Path(root)
    if not p.exists(): continue
    bad=0
    for f in p.rglob("*"):
        if f.is_file() and f.suffix in IMG_EXTS:
            try:
                Image.open(f).verify()
            except:
                print(f"CORRUPT: {f}")
                bad+=1
    print(f"Clean check {root}: corrupted={bad}")
print("clean_dataset.py DONE")
