# simulation/line_sim.py  (fotoğraflı + çerçeve renkli)
import os, random, time
import pygame, requests
from PIL import Image
import io

API_URL = "http://127.0.0.1:5000/analyze"
DATA_VAL = "data/dataset/val"

WIDTH, HEIGHT = 1000, 560
FPS = 60

BELT_Y = 300
BELT_SPEED = 0.9
SPAWN_MS = 2800
SCAN_X = 380
GATE_RED_X = 620
GATE_GREEN_X = 820
GATE_OPEN_MS = 900

POTATO_W, POTATO_H = 96, 96   # daha büyük ve net
DROP_ACCEL = 0.25

BG = (17,18,22)
BELT_DARK = (28,30,35)
BELT_LIGHT = (40,42,48)
WHITE = (235,235,240)
SILVER = (200,200,200)
GREEN = (40,200,90)
RED   = (220,60,60)
YELLOW= (255,225,80)
ORANGE= (255,170,60)
CYAN  = (80,200,220)
MUTED = (140,145,150)

def load_val_paths():
    cls_dirs = []
    for cls in ("healthy", "green", "rotten"):
        p = os.path.join(DATA_VAL, cls)
        if os.path.isdir(p):
            files = [os.path.join(p, f) for f in os.listdir(p)
                     if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))]
            if files:
                cls_dirs.append((cls, files))
    return cls_dirs

VAL_POOL = load_val_paths()
if not VAL_POOL:
    raise SystemExit("Val klasöründe görüntü yok.")

def _load_surface(path, size):
    """
    Görseli Pygame'e Surface olarak yükle. WEBP gibi formatlarda
    pygame takılırsa PIL ile dönüştürüp geçelim.
    """
    try:
        img = pygame.image.load(path).convert_alpha()
        img = pygame.transform.smoothscale(img, size)
        return img
    except Exception:
        with Image.open(path).convert("RGBA") as im:
            im = im.resize(size, Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            buf.seek(0)
            img = pygame.image.load(buf, "dummy.png").convert_alpha()
            return img

def pick_random_val_file():
    _, files = random.choice(VAL_POOL)
    return random.choice(files)

def call_api_with_file(path):
    with open(path, "rb") as f:
        r = requests.post(API_URL, files={"file": f}, timeout=6)
    r.raise_for_status()
    return r.json()

class Potato:
    def __init__(self, x, y, file_path):
        self.x, self.y = x, y
        self.w, self.h = POTATO_W, POTATO_H
        self.vy = 0.0
        self.file_path = file_path
        self.img = _load_surface(file_path, (self.w, self.h))
        self.scanned = False
        self.decided = False
        self.route = "MAIN"
        self.label = "PENDING"
        self.conf = 0.0
        self.low = False
        self.tie = False
        self.frame_color = MUTED
        self.tag = "PENDING"

    def decide_from_result(self, res: dict):
        self.label = res.get("label", "healthy")
        self.conf  = float(res.get("confidence", 0.0))
        self.low   = bool(res.get("low_confidence", False))
        self.tie   = bool(res.get("tie_green_healthy", False))

        if self.low or self.tie:
            self.route = "HOLD"
            self.frame_color = ORANGE if self.low else YELLOW
            self.tag   = "LOW" if self.low else "TIE"
        elif self.label == "rotten":
            self.route = "REJECT_RED"
            self.frame_color = RED
            self.tag = "ROTTEN"
        elif self.label == "green":
            self.route = "REJECT_GREEN"
            self.frame_color = GREEN
            self.tag = "GREEN"
        else:
            self.route = "MAIN"
            self.frame_color = SILVER
            self.tag = "HEALTHY"

        self.decided = True

def draw_belt(surface, t):
    surface.fill(BG)
    pygame.draw.rect(surface, BELT_DARK, (0, BELT_Y-16, WIDTH, 64))
    offset = int((t*BELT_SPEED*2) % 50)
    for x in range(-offset, WIDTH, 50):
        pygame.draw.rect(surface, BELT_LIGHT, (x, BELT_Y-14, 30, 60), border_radius=8)
    pygame.draw.line(surface, CYAN, (SCAN_X, BELT_Y-28), (SCAN_X, BELT_Y+52), 2)

def draw_gate(surface, x, open_until_ms, now_ms, color):
    pygame.draw.rect(surface, (60, 60, 65), (x-4, BELT_Y-40, 8, 80), border_radius=2)
    led_on = now_ms < open_until_ms
    pygame.draw.circle(surface, color if led_on else (100,100,105),
                       (x, BELT_Y+48), 9 if led_on else 7)

def draw_potato(surface, p: Potato, font):
    # Fotoğrafı çiz
    surface.blit(p.img, (p.x, p.y))
    # Çerçeve (sonuca göre renk)
    border_rect = pygame.Rect(p.x-4, p.y-4, p.w+8, p.h+8)
    pygame.draw.rect(surface, p.frame_color, border_rect, width=4, border_radius=10)
    # Etiket
    caption = f"{p.tag} {p.conf:.2f}" if p.decided else "PENDING"
    txt = font.render(caption, True, WHITE)
    surface.blit(txt, (p.x - 4, p.y + p.h + 6))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("PATATO BELT")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    potatoes = []
    last_spawn = pygame.time.get_ticks()
    gate_red_open_til = 0
    gate_green_open_til = 0

    # İlk patates
    potatoes.append(Potato(40, BELT_Y- (POTATO_H//2), pick_random_val_file()))

    running = True
    last_info = "Ready."
    while running:
        dt = clock.tick(FPS)
        now = pygame.time.get_ticks()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        # Yavaş spawn
        if now - last_spawn >= SPAWN_MS:
            potatoes.append(Potato(40, BELT_Y - (POTATO_H // 2), pick_random_val_file()))
            last_spawn = now


        # Hareket/karar
        for p in potatoes:
            p.x += BELT_SPEED

            # SCAN çizgisi → API çağrısı
            if (not p.scanned) and (p.x + p.w//2 >= SCAN_X):
                t0 = time.perf_counter()
                try:
                    res = call_api_with_file(p.file_path)
                    p.decide_from_result(res)
                    latency = (time.perf_counter() - t0)*1000
                    last_info = f"{os.path.basename(p.file_path)} -> {p.label} ({p.conf:.2f}) [{int(latency)} ms]"
                except Exception as ex:
                    p.decide_from_result({"label":"healthy","confidence":0.0,
                                          "low_confidence":True,"tie_green_healthy":False})
                    last_info = f"API error -> HOLD: {ex}"
                p.scanned = True

            # Kapak/düşüş
            if p.decided:
                if p.route == "REJECT_RED" and p.x + p.w//2 >= GATE_RED_X:
                    gate_red_open_til = now + GATE_OPEN_MS
                    p.vy += DROP_ACCEL
                    p.y += p.vy
                elif p.route == "REJECT_GREEN" and p.x + p.w//2 >= GATE_GREEN_X:
                    gate_green_open_til = now + GATE_OPEN_MS
                    p.vy += DROP_ACCEL
                    p.y += p.vy
                # HOLD/MAIN → düşüş yok

        # Çizimler
        t = pygame.time.get_ticks()/1000.0
        draw_belt(screen, t)
        draw_gate(screen, GATE_RED_X, gate_red_open_til, now, RED)
        draw_gate(screen, GATE_GREEN_X, gate_green_open_til, now, GREEN)

        removed = []
        for p in potatoes:
            draw_potato(screen, p, font)
            if p.x > WIDTH + 120 or p.y > HEIGHT + 120:
                removed.append(p)
        for p in removed:
            try: potatoes.remove(p)
            except: pass

        # HUD
        hud = [
            f"FPS: {clock.get_fps():.0f}",
            f"Info: {last_info}",
            "Frame Colors: low→ORANGE | tie→YELLOW | rotten→RED | green→GREEN | healthy→GRAY",
            "Speed/Flow: BELT_SPEED, SPAWN_MS; Cover time: GATE_OPEN_MS"
        ]
        for i, s in enumerate(hud):
            screen.blit(font.render(s, True, WHITE if i < 3 else (180,180,185)), (12, 14 + i*22))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
