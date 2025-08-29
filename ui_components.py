import pygame
from typing import Optional, Callable, List

# Colors
COL_BG = (20, 20, 26)
COL_PANEL = (36, 36, 48)
COL_ACC = (95, 175, 255)
COL_ACC2 = (120, 220, 160)
COL_TXT = (235, 235, 235)
COL_MUTED = (150, 150, 160)
COL_ERR = (240, 80, 80)
COL_OK = (120, 220, 120)
COL_GRID = (72, 72, 88)
COL_GRID_BAR = (96, 96, 120)
COL_CLIP = (85, 120, 200)
COL_CLIP_SEL = (180, 140, 80)
COL_FADE = (240, 240, 240)
COL_METER_BG = (40, 40, 55)
COL_METER = (90, 220, 120)
COL_MUTE = (210, 80, 80)
COL_SOLO = (240, 200, 90)
COL_SCROLL = (70, 70, 90)
COL_SCROLL_KNOB = (110, 110, 140)
COL_DELETE = (220, 80, 80)
COL_BPM_BG = (45, 45, 60)
COL_BPM_TEXT = (220, 220, 240)

# Fonts
pygame.font.init()
FONT_MD = pygame.font.SysFont("Arial", 16, bold=True)
FONT_SM = pygame.font.SysFont("Arial", 12)
FONT_LG = pygame.font.SysFont("Arial", 24, bold=True)

class Button:
    def __init__(self, x, y, w, h, label, fn, toggle=False, state=False, color=COL_ACC, hover_color=None, text_color=COL_TXT):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.fn = fn
        self.toggle = toggle
        self.state = state
        self.hover = False
        self.color = color
        self.text_color = text_color
        self.hover_color = hover_color or (min(color[0]+40, 255), min(color[1]+40, 255), min(color[2]+40, 255))

    def draw(self, surf):
        base = self.color if (self.toggle and self.state) else self.color
        col = base if not self.hover else self.hover_color
        pygame.draw.rect(surf, col, self.rect, border_radius=6)
        pygame.draw.rect(surf, (col[0]//2, col[1]//2, col[2]//2), self.rect, 2, border_radius=6)
        t = FONT_SM.render(self.label, True, self.text_color)
        surf.blit(t, t.get_rect(center=self.rect.center))

    def update(self, mouse):
        self.hover = self.rect.collidepoint(mouse)

    def handle_click(self):
        if self.toggle:
            self.state = not self.state
        if self.fn:
            self.fn()

class Slider:
    def __init__(self, x, y, w, label, minv, maxv, val, show_value=True):
        self.rect = pygame.Rect(x, y, w, 20)
        self.label = label
        self.min = minv
        self.max = maxv
        self.val = val
        self.drag = False
        self.show_value = show_value

    def draw(self, surf):
        if self.show_value:
            surf.blit(FONT_SM.render(f"{self.label}: {self.val:.2f}", True, COL_TXT), (self.rect.x, self.rect.y - 16))
        else:
            surf.blit(FONT_SM.render(f"{self.label}", True, COL_TXT), (self.rect.x, self.rect.y - 16))
        track = pygame.Rect(self.rect.x, self.rect.y + 9, self.rect.w, 3)
        pygame.draw.rect(surf, (70, 70, 90), track)
        rel = (self.val - self.min) / (self.max - self.min + 1e-12)
        x = self.rect.x + int(rel * self.rect.w)
        pygame.draw.circle(surf, COL_ACC, (x, self.rect.y + 10), 8)

    def update(self, mouse, pressed):
        if pressed[0]:
            if self.drag or self.rect.collidepoint(mouse):
                self.drag = True
                rel = (mouse[0] - self.rect.x) / max(1, self.rect.w)
                rel = max(0, min(1, rel))
                self.val = self.min + rel * (self.max - self.min)
        else:
            self.drag = False

class TextInput:
    def __init__(self, x, y, w, h, label, default_text="", numeric_only=False):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.text = default_text
        self.active = False
        self.numeric_only = numeric_only
        self.blink_timer = 0
        self.cursor_visible = True

    def draw(self, surf):
        # Draw label
        surf.blit(FONT_SM.render(self.label, True, COL_TXT), (self.rect.x, self.rect.y - 16))
        
        # Draw background
        bg_color = (50, 50, 70) if self.active else (40, 40, 55)
        pygame.draw.rect(surf, bg_color, self.rect, border_radius=4)
        pygame.draw.rect(surf, COL_ACC if self.active else (70, 70, 90), self.rect, 2, border_radius=4)
        
        # Draw text
        text_surf = FONT_MD.render(self.text, True, COL_TXT)
        surf.blit(text_surf, (self.rect.x + 5, self.rect.y + (self.rect.h - text_surf.get_height()) // 2))
        
        # Draw cursor if active
        if self.active and self.cursor_visible:
            cursor_x = self.rect.x + 5 + text_surf.get_width()
            pygame.draw.line(surf, COL_TXT, (cursor_x, self.rect.y + 5), 
                            (cursor_x, self.rect.y + self.rect.h - 5), 2)

    def update(self, events, dt):
        self.blink_timer += dt
        if self.blink_timer > 500:  # Blink every 500ms
            self.blink_timer = 0
            self.cursor_visible = not self.cursor_visible
            
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.active = self.rect.collidepoint(event.pos)
                
            if event.type == pygame.KEYDOWN and self.active:
                if event.key == pygame.K_RETURN:
                    self.active = False
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                elif event.key == pygame.K_ESCAPE:
                    self.active = False
                else:
                    if self.numeric_only:
                        if event.unicode.isdigit() or event.unicode == '.':
                            self.text += event.unicode
                    else:
                        self.text += event.unicode
                        
    def get_value(self):
        try:
            return float(self.text) if self.text else 0.0
        except ValueError:
            return 0.0