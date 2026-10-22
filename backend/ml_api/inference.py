import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import yaml

def get_model_arch(name, num_classes):
    name=name.lower()
    if "efficientnet" in name:
        m=models.efficientnet_b0(weights=None)
        in_f=m.classifier[1].in_features
        m.classifier[1]=torch.nn.Linear(in_f, num_classes)
        return m
    elif "resnet" in name:
        m=models.resnet50(weights=None)
        m.fc=torch.nn.Linear(m.fc.in_features, num_classes)
        return m
    else:
        m=models.efficientnet_b0(weights=None)
        m.classifier[1]=torch.nn.Linear(m.classifier[1].in_features, num_classes)
        return m

def load_model(path="models/cnn/efficientnet_b0_best.pth"):
    ckpt=torch.load(path, map_location="cpu")
    num_classes=int(ckpt.get("num_classes", 42))
    model_name=ckpt.get("model","efficientnet_b0")
    model=get_model_arch(model_name, num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# try to load class names from splits if available
LABELS=None
try:
    import pandas as pd
    if Path("data/processed/splits/train.csv").exists():
        df=pd.read_csv("data/processed/splits/train.csv")
        classes=sorted(df["label"].unique())
        LABELS={i:c for i,c in enumerate(classes)}
except:
    pass

def predict_image(model, pil_img):
    x=transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        logits=model(x)
        probs=F.softmax(logits, dim=1)[0]
        conf, pred = torch.max(probs, 0)
        top3 = torch.topk(probs, k=min(3, len(probs)))
        top_preds=[]
        for idx, p in zip(top3.indices.tolist(), top3.values.tolist()):
            label=LABELS.get(idx, f"class_{idx}") if LABELS else f"class_{idx}"
            # split label into crop/disease if contains _
            parts=label.split("_",1)
            crop=parts[0] if len(parts)>1 else label
            disease=parts[1] if len(parts)>1 else label
            top_preds.append({"class": label, "crop": crop, "disease": disease, "confidence": float(p), "class_id": idx})
        best=top_preds[0]
        return {
            "crop": best["crop"],
            "disease": best["disease"],
            "confidence": best["confidence"],
            "predicted_class": best["class"],
            "predicted_class_id": best["class_id"],
            "top_predictions": top_preds,
            "model": "efficientnet_b0",
            "explanation_available": True
        }
