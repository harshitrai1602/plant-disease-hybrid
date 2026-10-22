import argparse, os, yaml, time, json
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from torchvision import models
from sklearn.metrics import accuracy_score

def get_model(name, num_classes):
    name = name.lower()
    if name == "efficientnet_b0":
        try:
            from torchvision.models import EfficientNet_B0_Weights
            m = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        except:
            m = models.efficientnet_b0(pretrained=True)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
        return m
    elif name == "resnet50":
        try:
            from torchvision.models import ResNet50_Weights
            m = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        except:
            m = models.resnet50(pretrained=True)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    elif name in ["mobilenet_v3", "mobilenet_v3_large", "mobilenetv3"]:
        try:
            from torchvision.models import MobileNet_V3_Large_Weights
            m = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        except:
            m = models.mobilenet_v3_large(pretrained=True)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unknown model {name}")

class LeafDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = row["image_path"]
        label = int(row["class_id"])
        try:
            img = Image.open(path).convert("RGB")
        except:
            # return black image if corrupt
            img = Image.new("RGB", (224,224), (0,0,0))
        if self.transform:
            img = self.transform(img)
        return img, label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="efficientnet_b0", choices=["efficientnet_b0","resnet50","mobilenet_v3"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    # load config
    seed = 42
    if Path("configs/config.yaml").exists():
        cfg = yaml.safe_load(open("configs/config.yaml"))
        seed = cfg.get("seed", 42)

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {args.model} | Epochs: {args.epochs} | Batch: {args.batch_size}")

    train_df = pd.read_csv("data/processed/splits/train.csv")
    val_df = pd.read_csv("data/processed/splits/val.csv")
    num_classes = int(train_df["class_id"].nunique())
    print(f"Train {len(train_df)} Val {len(val_df)} Classes {num_classes}")

    # transforms - no augmentation on val
    train_tf = T.Compose([T.Resize((224,224)), T.RandomHorizontalFlip(p=0.5), T.RandomRotation(15), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    val_tf = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    # use subset for fast demo if dataset huge - comment out for full training
    # train_df = train_df.sample(min(2000, len(train_df)), random_state=seed)
    # val_df = val_df.sample(min(400, len(val_df)), random_state=seed)

    train_ds = LeafDataset(train_df, train_tf)
    val_ds = LeafDataset(val_df, val_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = get_model(args.model, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_acc = 0
    os.makedirs("models/cnn", exist_ok=True)
    os.makedirs("evaluation/reports", exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        running_loss = 0
        t0 = time.time()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_ds)

        # validation
        model.eval()
        val_loss = 0
        preds, targets = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                loss = criterion(out, labels)
                val_loss += loss.item() * imgs.size(0)
                preds.extend(out.argmax(1).cpu().tolist())
                targets.extend(labels.cpu().tolist())
        val_loss /= len(val_ds)
        val_acc = accuracy_score(targets, preds)
        scheduler.step(val_loss)
        print(f"Epoch {epoch}/{args.epochs} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f} | Time {time.time()-t0:.1f}s")

        # save best
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({"model": args.model, "state_dict": model.state_dict(), "num_classes": num_classes, "val_acc": val_acc}, f"models/cnn/{args.model}_best.pth")
            print(f"  -> Saved models/cnn/{args.model}_best.pth (best_acc={best_acc:.4f})")

    print(f"DONE. Best Val Acc: {best_acc:.4f}")
    # save log for thesis
    with open(f"evaluation/reports/{args.model}_training_log.json","w") as f:
        json.dump({"model": args.model, "epochs": args.epochs, "best_val_acc": best_acc, "status": "DEMO 1 EPOCH - RUN WITH --epochs 20 FOR FINAL"}, f, indent=2)

if __name__ == "__main__":
    main()
