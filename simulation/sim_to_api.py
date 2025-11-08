import os, time, threading, queue, random
import pygame
import requests
from pathlib import Path

# ---------------- Config ----------------
API_URL = "http://127.0.0.1:5000/analyze"  # Flask URL
IMG_DIRS = [
    "data/dataset/val/healthy",
    "data/dataset/val/green",
    "data/dataset/val/rotten",
]
CLASSES = ["healthy","green","rotten"]
CONF_THRESHOLD = 0.60  # altı 'LOW' diye işaretle
FPS = 60
SPEED = 2   # konveyör hızı (px/frame)

# --------------- Helpers ----------------
def load_images():
    imgs = []
    for d in IMG_DIRS:
        p = Path(d)
        if not p.exists(): 
            continue
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".webp"]:
                imgs.append(str(f))
    random.shuffle(imgs)
    return imgs

def analyze_file(path):
    try:
        with open(path, "rb") as f:
            r = requests.post(API_URL, files={"file": f}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# Arka planda inference kuyruğu (UI donmasın)
class Analyzer(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.q_in  = queue.Queue()
        self.q_out = queue.Queue()
    def run(self):
        while True:
            item_id, path = self.q_in.get()
            res = analyze_file(path)
            self.q_out.put((item_id, res))

# --------------- Pygame -----------------
W, H = 1000, 420
CAM_X = 480    # “kamera çizgisi”

class Potato:
    def __init__(self, path, x, y):
        self.path = path
        self.img  = pygame.image.load(path).convert()
        self.img  = pygame.transform.smoothscale(self.img, (120, 120))
        self.rect = self.img.get_rect()
        self.rect.topleft = (x, y)
        self.id   = id(self)
        self.sent = False
        self.label = "..."
        self.conf  = 0.0
        self.low_conf = False
        self.error = None

    def update(self):
        self.rect.x += SPEED

    def draw(self, screen, font):
        screen.blit(self.img, self.rect)
        # kutu ve metin
        text = f"{self.label} ({self.conf:.2f})" if self.error is None else f"ERR"
        color = (0,200,0) if self.label=="healthy" else (200,200,0) if self.label=="green" else (200,0,0)
        if self.low_conf: color = (255,140,0)
        pygame.draw.rect(screen, color, self.rect, 2)
        label_surf = font.render(text, True, color)
        screen.blit(label_surf, (self.rect.x, self.rect.y - 20))

def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Potato Conveyor — Pygame + Flask")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    # Arka plan / bant
    bg = pygame.Surface((W, H))
    bg.fill((30, 30, 30))
    belt = pygame.Rect(0, H//2 - 60, W, 120)

    # Kamera çizgisi
    cam_line = pygame.Rect(CAM_X, 0, 3, H)

    # Görselleri yükle
    files = load_images()
    if not files:
        print("Uyarı: Görsel bulunamadı. IMG_DIRS dizinlerini kontrol et.")
    # İlk nesneleri biraz geriden başlat
    items = []
    x = -200
    lanes = [belt.top + 5, belt.top + 55]  # iki hat
    for i, f in enumerate(files[:15]):  # fazla kalabalık olmasın
        items.append(Potato(f, x, random.choice(lanes)))
        x -= random.randint(160, 260)

    # Analyzer thread
    analyzer = Analyzer()
    analyzer.start()

    running = True
    t0 = time.time()

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    # yeni bir batch ekle
                    for f in random.sample(files, min(5, len(files))):
                        items.append(Potato(f, -200, random.choice(lanes)))

        # Update
        for it in items:
            it.update()
            # Kamera çizgisine ilk kez geçtiğinde kuyruğa at
            if not it.sent and it.rect.centerx >= CAM_X:
                analyzer.q_in.put((it.id, it.path))
                it.sent = True

        # Cevapları topla
        while not analyzer.q_out.empty():
            item_id, res = analyzer.q_out.get()
            for it in items:
                if it.id == item_id:
                    if "error" in res:
                        it.error = res["error"]
                        it.label = "error"
                        it.conf = 0.0
                        it.low_conf = True
                    else:
                        it.label = res.get("label","?")
                        it.conf  = float(res.get("confidence",0.0))
                        it.low_conf = (it.conf < CONF_THRESHOLD)
                    break

        # Çiz
        screen.blit(bg, (0,0))
        pygame.draw.rect(screen, (50, 50, 50), belt)           # konveyör
        pygame.draw.rect(screen, (120, 120, 255), cam_line)    # kamera çizgisi
        # bilgi paneli
        msg = f"Items: {len(items)}   API: {API_URL}   FPS: {int(clock.get_fps())}"
        screen.blit(font.render(msg, True, (200,200,200)), (10, 10))
        screen.blit(font.render("Space: yeni batch, Esc: çıkış", True, (160,160,160)), (10, 34))

        for it in items:
            it.draw(screen, font)

        # Ekranın dışına çıkanları at
        items = [it for it in items if it.rect.left <= W + 40]

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
