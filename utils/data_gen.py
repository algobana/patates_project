import os, shutil, random
from pathlib import Path

CLASSES = ["healthy", "green", "rotten"]

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def split_dataset(raw_dir: str, out_dir: str, train=0.7, val=0.15, test=0.15, seed=42):
    assert abs(train + val + test - 1.0) < 1e-9, "train+val+test = 1 olmalı"
    random.seed(seed)
    raw = Path(raw_dir)
    out = Path(out_dir)

    # hedef klasörleri oluştur
    for split in ["train", "val", "test"]:
        for c in CLASSES:
            safe_mkdir(out / split / c)

    # her sınıfı ayrı ayrı böl
    for c in CLASSES:
        src = raw / c
        imgs = [p for p in src.glob("*.*") if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".webp"]]
        if not imgs:
            print(f"[UYARI] {src} boş görünüyor.")
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * train)
        n_val   = int(n * val)
        train_files = imgs[:n_train]
        val_files   = imgs[n_train:n_train+n_val]
        test_files  = imgs[n_train+n_val:]

        for dst_split, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            for f in files:
                shutil.copy2(f, out / dst_split / c / f.name)

        print(f"[OK] {c}: total={n} -> train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

if __name__ == "__main__":
    split_dataset("data/raw", "data/dataset")
