from pathlib import Path
from PIL import Image

def quick_image_check(root="data/dataset/train", limit=5):
    root = Path(root)
    cnt = 0
    for p in root.rglob("*.*"):
        if p.suffix.lower() not in [".jpg",".jpeg",".png",".bmp",".webp"]:
            continue
        try:
            Image.open(p).convert("RGB")
            cnt += 1
            if cnt >= limit:
                break
        except Exception as e:
            print(f"Bozuk dosya: {p} -> {e}")
    print(f"Açılan örnek sayısı: {cnt}")

if __name__ == "__main__":
    quick_image_check()
