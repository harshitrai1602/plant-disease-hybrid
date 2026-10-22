import zipfile
from pathlib import Path
import shutil

zip_path = Path("data/raw/plantdoc.zip")
out_dir = Path("data/raw/plantdoc_extract")

if not zip_path.exists():
    print("ERROR: plantdoc.zip not found")
    raise SystemExit(1)

if out_dir.exists():
    shutil.rmtree(out_dir)

out_dir.mkdir(parents=True, exist_ok=True)

print("ZIP:", zip_path)
print("Size: %.2f GB" % (zip_path.stat().st_size / (1024**3)))

extracted = 0
skipped = 0

with zipfile.ZipFile(zip_path, "r") as z:
    entries = z.infolist()

    image_entries = [
        e for e in entries
        if Path(e.filename).suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    print("Image entries in ZIP:", len(image_entries))

    for i, entry in enumerate(image_entries, 1):

        # Skip the documentation/example image
        if Path(entry.filename).name.lower() == "plantdoc_examples.png":
            skipped += 1
            continue

        # Convert ZIP separators to Windows-safe path structure
        parts = [
            p for p in entry.filename.replace("\\", "/").split("/")
            if p not in ("", ".", "..")
        ]

        if not parts:
            skipped += 1
            continue

        # Remove the repository root
        if parts[0].lower().startswith("plantdoc-dataset"):
            parts = parts[1:]

        # Remove characters that Windows does not allow in filenames
        safe_parts = []

        for p in parts:
            p = p.replace(":", "_")
            p = p.replace("*", "_")
            p = p.replace("?", "_")
            p = p.replace('"', "_")
            p = p.replace("<", "_")
            p = p.replace(">", "_")
            p = p.replace("|", "_")

            # Windows also dislikes trailing spaces/dots
            p = p.rstrip(" .")

            if not p:
                p = "_"

            safe_parts.append(p)

        destination = out_dir.joinpath(*safe_parts)

        try:
            destination.parent.mkdir(parents=True, exist_ok=True)

            with z.open(entry) as source, open(destination, "wb") as target:
                shutil.copyfileobj(source, target)

            extracted += 1

        except Exception as ex:
            skipped += 1
            print("SKIPPED:", entry.filename)
            print("REASON:", ex)

        if i % 250 == 0:
            print(f"Processed {i}/{len(image_entries)}")

print()
print("========== EXTRACTION COMPLETE ==========")
print("Extracted:", extracted)
print("Skipped:", skipped)

images = [
    p for p in out_dir.rglob("*")
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
]

print("Images physically present:", len(images))

print()
print("Sample images:")

for p in images[:20]:
    print(p)
