import os, json, csv
from pathlib import Path
from PIL import Image
from collections import Counter
import numpy as np

RAW_PV = Path("data/raw/plantvillage")
RAW_PD = Path("data/raw/plantdoc")
OUT_JSON = Path("evaluation/reports/dataset_summary.json")
OUT_CSV = Path("evaluation/reports/dataset_statistics.csv")
OUT_PROV = Path("evaluation/reports/image_provenance.csv")
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".JPG",".JPEG",".PNG"}

def scan_dataset(path):
    files=[]
    if not path.exists():
        return files
    for p in path.rglob("*"):
        if p.is_file() and p.suffix in IMG_EXTS:
            files.append(p)
    return files

def is_corrupt(p):
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            im.load()
        return False
    except:
        return True

os.makedirs("evaluation/reports", exist_ok=True)
pv_files = scan_dataset(RAW_PV)
pd_files = scan_dataset(RAW_PD)
all_files = [("plantvillage", f) for f in pv_files] + [("plantdoc", f) for f in pd_files]
total = len(all_files)
corrupted = 0
formats = Counter()
sizes = []
per_class = Counter()
per_dataset = Counter({"plantvillage": len(pv_files), "plantdoc": len(pd_files)})
provenance_rows = []

for dataset, f in all_files:
    try:
        base = RAW_PV if dataset=="plantvillage" else RAW_PD
        rel = f.relative_to(base)
        cls = rel.parts[0] if len(rel.parts)>=1 else f.parent.name
    except:
        cls = f.parent.name
    per_class[f"{dataset}/{cls}"] += 1
    if is_corrupt(f):
        corrupted += 1
        continue
    try:
        with Image.open(f) as im:
            w,h = im.size
            fmt = im.format
            sizes.append((w,h))
            formats[str(fmt)] += 1
    except:
        pass
    provenance_rows.append([dataset, str(f), "", cls, "", str(f.name)])

if sizes:
    ws, hs = zip(*sizes)
    dim_stats = {"min_w":int(min(ws)), "max_w":int(max(ws)), "min_h":int(min(hs)), "max_h":int(max(hs)), "avg_w":float(np.mean(ws)), "avg_h":float(np.mean(hs))}
else:
    dim_stats = {"min_w":0, "max_w":0, "min_h":0, "max_h":0, "avg_w":0, "avg_h":0}

summary = {
    "status": "DATASET AVAILABLE" if total>0 else "DATASET NOT AVAILABLE",
    "total_images": total,
    "total_classes_raw": len(per_class),
    "images_per_dataset": dict(per_dataset),
    "images_per_class": dict(per_class),
    "corrupted_images": corrupted,
    "image_formats": dict(formats),
    "image_dimensions": dim_stats,
    "class_imbalance_ratio": round(max(per_class.values())/max(1,min(per_class.values())),2) if per_class else 0
}
with open(OUT_JSON,"w") as jf:
    json.dump(summary, jf, indent=2)
with open(OUT_CSV,"w",newline="") as cf:
    w=csv.writer(cf)
    w.writerow(["class","count","dataset"])
    for k,v in sorted(per_class.items()):
        ds = k.split("/")[0]
        w.writerow([k,v,ds])
with open(OUT_PROV,"w",newline="") as pf:
    w=csv.writer(pf)
    w.writerow(["source_dataset","source_image","annotation_id","class","bbox","processed_image"])
    for r in provenance_rows:
        w.writerow(r)
print(json.dumps(summary, indent=2))
print(f"Saved {OUT_JSON}")
