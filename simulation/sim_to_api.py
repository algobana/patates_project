# simulation/line_sim.py (FINAL PRO UI: RIGHT ALIGNED CONTROLS 📐✨)
import os, random, time
import pygame, requests
from PIL import Image
import io

# ================================
#  AYARLAR
# ================================
API_URL = "http://127.0.0.1:5000/analyze"
DATA_VAL = "data/dataset/val"
ANOMALY_DIR = "data/anomaly"

WIDTH, HEIGHT = 1000, 600
FPS = 60

BELT_Y = 250       # Bant konumu
DEFAULT_SPEED = 0.9 
SPAWN_MS = 2800
SCAN_X = 380
GATE_RED_X = 620   
GATE_GREEN_X = 820 
GATE_OPEN_MS = 900

POTATO_W, POTATO_H = 96, 96 
DROP_ACCEL = 0.25

# Renkler
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

# ================================
#  VERİ YÜKLEME
# ================================
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
    print("UYARI: Val klasöründe görüntü yok.")

ANOMALY_POOL = []
if os.path.exists(ANOMALY_DIR):
    ANOMALY_POOL = [os.path.join(ANOMALY_DIR, f) for f in os.listdir(ANOMALY_DIR)
                    if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))]
    print(f"Anomali dosyaları yüklendi: {len(ANOMALY_POOL)} adet")

# ================================
#  YARDIMCI FONKSİYONLAR
# ================================
def _load_surface(path, size):
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

def pick_next_item(force_anomaly=False):
    if force_anomaly and ANOMALY_POOL:
        return random.choice(ANOMALY_POOL), True
    if ANOMALY_POOL and random.randint(0, 100) < 10:
        file_path = random.choice(ANOMALY_POOL)
        return file_path, True
    if VAL_POOL:
        _, files = random.choice(VAL_POOL)
        return random.choice(files), False
    return None, False

def call_api_with_file(path):
    with open(path, "rb") as f:
        r = requests.post(API_URL, files={"file": f}, timeout=6)
    r.raise_for_status()
    return r.json()

# ================================
#  POTATO CLASS
# ================================
class Potato:
    def __init__(self, x, y, file_path, is_anomaly=False):
        self.x, self.y = x, y
        self.w, self.h = POTATO_W, POTATO_H
        self.vy = 0.0
        self.file_path = file_path
        self.is_anomaly = is_anomaly
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
            self.route = "REJECT_RED"
            self.frame_color = ORANGE if self.low else YELLOW
            self.tag   = "RISK!" if self.low else "TIE"
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

# ================================
#  ÇİZİM FONKSİYONLARI (MERKEZİ)
# ================================
def draw_scene(screen, t, speed, gate_red_open, gate_green_open, now):
    """Arka plan, bant ve kapıları çizer"""
    screen.fill(BG)
    
    # Bant
    pygame.draw.rect(screen, BELT_DARK, (0, BELT_Y-16, WIDTH, 64))
    offset = int((t*speed*2) % 50) 
    for x in range(-offset, WIDTH, 50):
        pygame.draw.rect(screen, BELT_LIGHT, (x, BELT_Y-14, 30, 60), border_radius=8)
    pygame.draw.line(screen, CYAN, (SCAN_X, BELT_Y-28), (SCAN_X, BELT_Y+52), 2)

    # Kapılar
    # Red Gate
    pygame.draw.rect(screen, (60, 60, 65), (GATE_RED_X-4, BELT_Y-40, 8, 80), border_radius=2)
    led_on = now < gate_red_open
    pygame.draw.circle(screen, RED if led_on else (100,100,105), (GATE_RED_X, BELT_Y+48), 9 if led_on else 7)
    
    # Green Gate
    pygame.draw.rect(screen, (60, 60, 65), (GATE_GREEN_X-4, BELT_Y-40, 8, 80), border_radius=2)
    led_on = now < gate_green_open
    pygame.draw.circle(screen, GREEN if led_on else (100,100,105), (GATE_GREEN_X, BELT_Y+48), 9 if led_on else 7)

def draw_hud(screen, font, font_small, stats, last_info, fps, belt_speed):
    """3 BÖLGELİ HUD: SOL ÜST, SAĞ ÜST (Sağa Dayalı), SOL ALT"""
    
    # 1. Verimlilik Hesabı
    yield_rate = 0.0
    if stats["total"] > 0:
        yield_rate = (stats["healthy"] / stats["total"]) * 100

    # -----------------------------------------------
    # BÖLGE 1: SOL ÜST (İSTATİSTİKLER)
    # -----------------------------------------------
    top_stats = [
        f"FPS: {fps:.0f} | SPEED: x{belt_speed:.1f}",
        "--- PRODUCTION STATS ---",
        f"📦 TOTAL     : {stats['total']}",
        f"✅ ACCEPTED  : {stats['healthy']}",
        f"❌ REJECTED  : {stats['rotten'] + stats['green'] + stats['risk']}",
        f"   - Rotten  : {stats['rotten']}",
        f"   - Green   : {stats['green']}",
        f"   - Stone   : {stats['risk']}",
        f"📈 YIELD     : %{yield_rate:.1f}",
    ]
    
    for i, s in enumerate(top_stats):
        color = WHITE
        if "YIELD" in s: color = GREEN if yield_rate > 80 else YELLOW
        screen.blit(font_small.render(s, True, color), (12, 10 + i*20))

    # -----------------------------------------------
    # BÖLGE 2: SAĞ ÜST (OPERATOR CONTROLS) -> SAĞA YAPIŞIK
    # -----------------------------------------------
    controls_info = [
        "--- OPERATOR CONTROLS ---",
        "PAUSE/RESUME [SPACE]", # Tuşları sona aldım ki daha düzgün dursun
        "FORCE STONE [S]",
        "ADJUST SPEED [UP/DN]",
        "RESET STATS [R]"
    ]
    
    right_margin = 12 # Sağdan boşluk
    
    for i, s in enumerate(controls_info):
        txt_surf = font.render(s, True, SILVER)
        txt_w = txt_surf.get_width()
        # Ekran Genişliği - Kenar Boşluğu - Yazı Genişliği = Başlangıç X
        x_pos = WIDTH - right_margin - txt_w 
        screen.blit(txt_surf, (x_pos, 10 + i*20))

    # -----------------------------------------------
    # BÖLGE 3: SOL ALT (STATUS & LOGIC)
    # -----------------------------------------------
    bottom_y_start = 400
    
    logic_info = [
        f"STATUS: {last_info}",
        " ",
        "--- SORTING LOGIC ---",
        "RISKY/STONE -> REJECT (RED GATE)",
        "ROTTEN      -> REJECT (RED GATE)",
        "GREEN       -> REJECT (GREEN GATE)",
        "HEALTHY     -> MAIN LINE"
    ]

    for i, s in enumerate(logic_info):
        color = WHITE
        if "STATUS" in s:
            if "SAFETY STOP" in s: color = ORANGE
            elif "PAUSED" in s: color = YELLOW
            elif "Scanning" in s: color = CYAN
        elif "SORTING" in s: color = SILVER
        elif "RISKY" in s: color = ORANGE
        elif "ROTTEN" in s: color = RED
        elif "GREEN" in s and "GATE" in s: color = GREEN
        
        screen.blit(font.render(s, True, color), (12, bottom_y_start + i*20))

def flush_system():
    return "System Flushed. Ready."

# ================================
#  MAIN LOOP
# ================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AI POTATO SORTING SYSTEM - ULTIMATE EDITION")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    font_small = pygame.font.SysFont("consolas", 16)
    
    # --- KONTROL DEĞİŞKENLERİ ---
    belt_speed = DEFAULT_SPEED
    paused = False
    pause_start_time = 0 
    force_next_anomaly = False

    stats = {"total": 0, "healthy": 0, "rotten": 0, "green": 0, "risk": 0}
    potatoes = []
    last_spawn = pygame.time.get_ticks()
    gate_red_open_til = 0
    gate_green_open_til = 0

    path, is_anom = pick_next_item()
    if path:
        potatoes.append(Potato(40, BELT_Y- (POTATO_H//2), path, is_anom))

    running = True
    last_info = "System Ready."
    in_reject_mode = False 

    while running:
        dt = clock.tick(FPS)
        now = pygame.time.get_ticks()
        
        # --- EVENT HANDLING ---
        for e in pygame.event.get():
            if e.type == pygame.QUIT: running = False
            
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:  # SMART PAUSE
                    paused = not paused
                    if paused:
                        last_info = "PAUSED (Manual Stop)"
                        pause_start_time = now
                    else:
                        last_info = "Resuming..."
                        pause_duration = now - pause_start_time
                        last_spawn += pause_duration
                
                elif e.key == pygame.K_s:    # SPAWN STONE
                    force_next_anomaly = True
                    last_info = "CMD: Spawn Stone Next!"
                
                elif e.key == pygame.K_UP:   # SPEED UP
                    belt_speed += 0.2
                    last_info = f"SPEED UP: {belt_speed:.1f}"
                
                elif e.key == pygame.K_DOWN: # SPEED DOWN
                    belt_speed = max(0.2, belt_speed - 0.2)
                    last_info = f"SPEED DOWN: {belt_speed:.1f}"
                
                elif e.key == pygame.K_r:    # RESET STATS
                    stats = {k:0 for k in stats}
                    last_info = "STATS RESET."

        # AKIŞ
        if not paused:
            current_spawn_ms = SPAWN_MS / (belt_speed if belt_speed > 0 else 1)
            
            if now - last_spawn >= current_spawn_ms:
                path, is_anom = pick_next_item(force_anomaly=force_next_anomaly)
                if force_next_anomaly: force_next_anomaly = False
                
                if path:
                    potatoes.append(Potato(40, BELT_Y - (POTATO_H // 2), path, is_anom))
                last_spawn = now

            for p in potatoes:
                p.x += belt_speed 

                # SCANNING (Anti-Flicker)
                if (not p.scanned) and (p.x + p.w//2 >= SCAN_X):
                    last_info = "Scanning Object..." 
                    p.frame_color = CYAN 
                    
                    # 1. Çizim (Yazılar dahil)
                    draw_scene(screen, pygame.time.get_ticks()/1000.0, belt_speed, 
                               gate_red_open_til, gate_green_open_til, now)
                    
                    for p_draw in potatoes:
                        if p_draw == p: pass 
                        border_rect = pygame.Rect(p_draw.x-4, p_draw.y-4, p_draw.w+8, p_draw.h+8)
                        screen.blit(p_draw.img, (p_draw.x, p_draw.y))
                        pygame.draw.rect(screen, p_draw.frame_color, border_rect, width=4, border_radius=10)
                        
                        caption = f"{p_draw.tag} {p_draw.conf:.2f}" if p_draw.decided else ""
                        if p_draw.decided and p_draw.frame_color == ORANGE: caption = "REJECT!" 
                        if p_draw == p: caption = "SCAN..."
                        
                        if caption:
                            txt = font.render(caption, True, WHITE)
                            screen.blit(txt, (p_draw.x - 4, p_draw.y + p_draw.h + 6))

                    draw_hud(screen, font, font_small, stats, last_info, clock.get_fps(), belt_speed)
                    pygame.display.flip() 

                    # 2. API Çağrısı
                    t0 = time.perf_counter()
                    try:
                        res = call_api_with_file(p.file_path)
                        p.decide_from_result(res)
                        stats["total"] += 1
                        if p.tag == "HEALTHY": stats["healthy"] += 1
                        elif p.tag == "ROTTEN": stats["rotten"] += 1
                        elif p.tag == "GREEN": stats["green"] += 1
                        elif p.tag in ["RISK!", "TIE"]: stats["risk"] += 1
                        
                        latency = (time.perf_counter() - t0)*1000
                        unc_val = res.get('uncertainty', 0.0)
                        
                        if p.frame_color == ORANGE:
                            last_info = f"SAFETY STOP! (Unc:{unc_val:.2f}) -> REJECTING"
                            in_reject_mode = True
                        else:
                            if in_reject_mode:
                                print(flush_system())
                                in_reject_mode = False
                            last_info = f"Result: {p.label} (Conf:{p.conf:.2f}) [LAT:{int(latency)}ms]"
                    except Exception as ex:
                        p.decide_from_result({"label":"healthy","confidence":0.0,"low_confidence":True})
                        last_info = f"API FAIL -> REJECTING"
                    p.scanned = True

                # Drop
                if p.decided:
                    if p.route == "REJECT_RED" and p.x + p.w//2 >= GATE_RED_X:
                        gate_red_open_til = now + GATE_OPEN_MS
                        p.vy += DROP_ACCEL
                        p.y += p.vy
                    elif p.route == "REJECT_GREEN" and p.x + p.w//2 >= GATE_GREEN_X:
                        gate_green_open_til = now + GATE_OPEN_MS
                        p.vy += DROP_ACCEL
                        p.y += p.vy

            removed = []
            for p in potatoes:
                if p.x > WIDTH + 120 or p.y > HEIGHT + 120: removed.append(p)
            for p in removed:
                try: potatoes.remove(p)
                except: pass

        # --- NORMAL ÇİZİM ---
        draw_scene(screen, pygame.time.get_ticks()/1000.0, belt_speed, 
                   gate_red_open_til, gate_green_open_til, now)

        for p in potatoes:
            border_rect = pygame.Rect(p.x-4, p.y-4, p.w+8, p.h+8)
            screen.blit(p.img, (p.x, p.y))
            pygame.draw.rect(screen, p.frame_color, border_rect, width=4, border_radius=10)
            
            caption = f"{p.tag} {p.conf:.2f}" if p.decided else ""
            if p.decided and p.frame_color == ORANGE: caption = "REJECT!" 
            if not p.decided and p.x > SCAN_X - 10 and p.x < SCAN_X + 50: caption = "SCAN..."
            
            if caption:
                txt = font.render(caption, True, WHITE)
                screen.blit(txt, (p.x - 4, p.y + p.h + 6))

        draw_hud(screen, font, font_small, stats, last_info, clock.get_fps(), belt_speed)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()