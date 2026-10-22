from pathlib import Path
import pandas as pd
from PIL import Image
import hashlib

ROOT = Path("data/raw/plantdoc_extract")
OUT = Path("data/processed/metadata")
OUT.mkdir(parents=True, exist_ok=True)

VALID_EXT = {".jpg", ".jpeg", ".png"}

rows = []
invalid = []
seen_hashes = set()
duplicates = []

images = [
    p for p in ROOT.rglob("*")
    if p.is_file() and p.suffix.lower() in VALID_EXT
]

print("Images discovered:", len(images))

for i, path in enumerate(images, 1):

    rel = path.relative_to(ROOT)
    parts = rel.parts

    # Expected:
    # train/class/image
    # test/class/image
    if len(parts) < 3:
        continue

    split_source = parts[0]
    raw_class = parts[1]

    try:
        with Image.open(path) as img:
            img.verify()

        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()

        if digest in seen_hashes:
            duplicates.append(str(path))
            continue

        seen_hashes.add(digest)

        rows.append({
            "image_path": str(path.resolve()),
            "source_dataset": "plantdoc",
            "source_split": split_source,
            "raw_class": raw_class
        })

    except Exception as e:
        invalid.append({
            "image_path": str(path),
            "error": str(e)
        })

    if i % 250 == 0:
        print(f"Processed {i}/{len(images)}")

df = pd.DataFrame(rows)

df.to_csv(
    OUT / "plantdoc_manifest.csv",
    index=False,
    encoding="utf-8"
)

pd.DataFrame(invalid).to_csv(
    OUT / "plantdoc_invalid.csv",
    index=False,
    encoding="utf-8"
)

pd.DataFrame({
    "image_path": duplicates
}).to_csv(
    OUT / "plantdoc_duplicates.csv",
    index=False,
    encoding="utf-8"
)

print()
print("============================================")
print("PLANTDOC MANIFEST COMPLETE")
print("============================================")
print("Valid unique images :", len(df))
print("Invalid images      :", len(invalid))
print("Duplicate images    :", len(duplicates))

print()
print("CLASS COUNTS")
print(
    df.groupby(["source_split", "raw_class"])
      .size()
      .reset_index(name="count")
      .sort_values("count", ascending=False)
      .to_string(index=False)
)

print()
print("Created:")
print(OUT / "plantdoc_manifest.csv")
print(OUT / "plantdoc_invalid.csv")
print(OUT / "plantdoc_duplicates.csv")
