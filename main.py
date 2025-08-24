import os
import sys
import wave
import sqlite3
import warnings
import subprocess
from datetime import datetime

import numpy as np
import pygame

# ===== Optional libs (no DawDreamer) =====
try:
    import librosa
    import soundfile as sf  # noqa: F401
    LIBROSA_AVAILABLE = True
except Exception:
    LIBROSA_AVAILABLE = False

SR = 44100

warnings.filterwarnings("ignore", message=r".*__audioread_load.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=r"PySoundFile failed.*")

# ===== macOS-safe file dialogs =====
TK_AVAILABLE = False
if sys.platform != "darwin":
    try:
        import tkinter as tk
        from tkinter import filedialog
        TK_AVAILABLE = True
    except Exception:
        TK_AVAILABLE = False

def choose_open_file(title="Open", pattern="public.audio"):
    if sys.platform == "darwin":
        try:
            osa = f'POSIX path of (choose file with prompt "{title}" of type {{"{pattern}"}})'
            res = subprocess.run(["osascript", "-e", osa], capture_output=True, text=True)
            out = res.stdout.strip()
            return out if res.returncode == 0 and out else None
        except Exception:
            return None
    if TK_AVAILABLE:
        try:
            root = tk.Tk(); root.withdraw(); root.update()
            fp = filedialog.askopenfilename(title=title,
                                            filetypes=[("Audio", "*.wav *.mp3 *.flac *.aiff *.aif *.ogg"),
                                                       ("All files", "*.*")])
            root.destroy()
            return fp if fp else None
        except Exception:
            return None
    return None

def choose_save_file(title="Save As", default_name="mix.wav"):
    if sys.platform == "darwin":
        try:
            osa = f'POSIX path of (choose file name with prompt "{title}" default name "{default_name}")'
            res = subprocess.run(["osascript", "-e", osa], capture_output=True, text=True)
            path = res.stdout.strip() if res.returncode == 0 else None
            if path and not os.path.splitext(path)[1]:
                path += ".wav"
            return path
        except Exception:
            return None
    if TK_AVAILABLE:
        try:
            root = tk.Tk(); root.withdraw(); root.update()
            fp = filedialog.asksaveasfilename(title=title, defaultextension=".wav",
                                              filetypes=[("WAV", "*.wav")], initialfile=default_name)
            root.destroy()
            return fp if fp else None
        except Exception:
            return None
    return None

# ===== Audio helpers =====
def load_audio(path, target_sr=SR):
    if LIBROSA_AVAILABLE:
        y, _ = librosa.load(path, sr=target_sr, mono=True)
        y = y.astype(np.float32)
        m = float(np.max(np.abs(y))) if y.size else 0.0
        if m > 0:
            y = y / m
        return y, target_sr
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        ch = wf.getnchannels()
        data = wf.readframes(n)
        arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        if ch > 1:
            arr = arr.reshape(-1, ch).mean(axis=1)
        if sr != target_sr and arr.size:
            t_old = np.linspace(0, len(arr)/sr, len(arr), endpoint=False)
            t_new = np.linspace(0, len(arr)/sr, int(len(arr)*target_sr/sr), endpoint=False)
            arr = np.interp(t_new, t_old, arr).astype(np.float32)
            sr = target_sr
        return arr, sr

def time_stretch(y, speed):
    speed = max(0.25, min(2.0, float(speed)))
    if not y.size:
        return y.astype(np.float32)
    if LIBROSA_AVAILABLE:
        return librosa.effects.time_stretch(y, rate=speed).astype(np.float32)
    # naive resample (changes pitch)
    n_new = int(len(y) / speed)
    if n_new <= 0:
        n_new = 1
    t_old = np.linspace(0, 1, len(y), endpoint=False)
    t_new = np.linspace(0, 1, n_new, endpoint=False)
    return np.interp(t_new, t_old, y).astype(np.float32)

def to_stereo(mono):
    return np.vstack([mono, mono])

def to_int16_stereo(st):
    st = np.clip(st, -1.0, 1.0)
    return (st * 32767).astype(np.int16)

# ===== Data models =====
class Clip:
    def __init__(self, audio, name, start=0.0, speed=1.0, gain=1.0, in_pos=0.0, out_pos=None):
        self.audio = audio.astype(np.float32)  # mono
        self.name = name
        self.start = float(start)  # timeline start (sec)
        self.speed = float(speed)  # 1.0 normal; 0.5 slower; 2.0 faster
        self.gain = float(gain)
        self.in_pos = float(in_pos)  # seconds into original audio
        self.out_pos = float(out_pos) if out_pos is not None else (len(audio)/SR)
        self.track_index = 0
        self.selected = False
        # layout cache
        self.rect = None

    @property
    def src_len_sec(self):
        return len(self.audio) / SR

    @property
    def eff_len_sec(self):
        return max(0.0, (self.out_pos - self.in_pos) / max(self.speed, 1e-6))

    def bounds(self):
        return self.start, self.start + self.eff_len_sec

    def cut_at(self, t_sec):
        """Split clip at absolute timeline time t_sec; returns right part or None."""
        left, right = self.bounds()
        if t_sec <= left + 1e-6 or t_sec >= right - 1e-6:
            return None
        # timeline delta to split
        dt = t_sec - self.start
        # Source seconds consumed = dt * speed
        consumed = dt * self.speed
        split_src = self.in_pos + consumed
        # left remains [in_pos, split_src], right becomes [split_src, out_pos] starting at t_sec
        right_clip = Clip(
            audio=self.audio,
            name=self.name + " (split)",
            start=t_sec,
            speed=self.speed,
            gain=self.gain,
            in_pos=split_src,
            out_pos=self.out_pos,
        )
        # shorten left
        self.out_pos = split_src
        return right_clip

class Track:
    def __init__(self, name, volume=1.0, mute=False, solo=False):
        self.name = name
        self.volume = float(volume)
        self.mute = mute
        self.solo = solo
        self.clips = []

# ===== UI =====
pygame.init()
pygame.mixer.init(frequency=SR, size=-16, channels=2, buffer=1024)

WIDTH, HEIGHT = 1200, 820
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Audio Editor (No DawDreamer)")

COL_BG = (20,20,26)
COL_PANEL = (36,36,48)
COL_ACC = (95,175,255)
COL_ACC2 = (120,220,160)
COL_TXT = (235,235,235)
COL_MUTED = (150,150,160)
COL_ERR = (240,80,80)
COL_OK = (120,220,120)
COL_GRID = (70,70,85)
COL_CLIP = (85,120,200)
COL_CLIP_SEL = (180,140,80)

pygame.font.init()
FONT_LG = pygame.font.SysFont("Arial", 24, bold=True)
FONT_MD = pygame.font.SysFont("Arial", 16, bold=True)
FONT_SM = pygame.font.SysFont("Arial", 12)

class Button:
    def __init__(self, x,y,w,h,label,fn, toggle=False, state=False):
        self.rect = pygame.Rect(x,y,w,h)
        self.label = label
        self.fn = fn
        self.toggle = toggle
        self.state = state
        self.hover = False
    def draw(self, surf):
        base = COL_ACC2 if (self.toggle and self.state) else COL_ACC
        col = base if not self.hover else (min(base[0]+40,255),min(base[1]+40,255),min(base[2]+40,255))
        pygame.draw.rect(surf, col, self.rect, border_radius=6)
        pygame.draw.rect(surf, (col[0]//2,col[1]//2,col[2]//2), self.rect, 2, border_radius=6)
        t = FONT_SM.render(self.label, True, COL_TXT)
        surf.blit(t, t.get_rect(center=self.rect.center))
    def update(self, mouse):
        self.hover = self.rect.collidepoint(mouse)
    def handle(self, e):
        if e.type==pygame.MOUSEBUTTONDOWN and e.button==1 and self.hover:
            if self.toggle:
                self.state = not self.state
            if self.fn: self.fn()

class Slider:
    def __init__(self,x,y,w,label,minv,maxv,val):
        self.rect = pygame.Rect(x,y,w,20)
        self.label = label
        self.min=minv; self.max=maxv; self.val=val
        self.drag=False
    def draw(self, surf):
        surf.blit(FONT_SM.render(f"{self.label}: {self.val:.2f}", True, COL_TXT),(self.rect.x, self.rect.y-16))
        track = pygame.Rect(self.rect.x, self.rect.y+9, self.rect.w, 3)
        pygame.draw.rect(surf, (70,70,90), track)
        rel = (self.val-self.min)/(self.max-self.min)
        x = self.rect.x + int(rel*self.rect.w)
        pygame.draw.circle(surf, COL_ACC, (x, self.rect.y+10), 8)
    def update(self, mouse, pressed):
        if pressed[0]:
            if self.drag or self.rect.collidepoint(mouse):
                self.drag=True
                rel = (mouse[0]-self.rect.x)/self.rect.w
                rel = max(0,min(1,rel))
                self.val = self.min + rel*(self.max-self.min)
        else:
            self.drag=False

class TextInput:
    def __init__(self,x,y,w,label):
        self.rect=pygame.Rect(x,y,w,26); self.label=label; self.text=""; self.active=False
    def draw(self,surf):
        surf.blit(FONT_SM.render(self.label, True, COL_TXT),(self.rect.x, self.rect.y-16))
        pygame.draw.rect(surf, (100,140,200) if self.active else COL_ACC, self.rect, border_radius=6)
        pygame.draw.rect(surf, (60,90,140), self.rect, 2, border_radius=6)
        t=FONT_SM.render(self.text, True, COL_TXT); surf.blit(t,(self.rect.x+6,self.rect.y+5))
    def handle(self,e):
        if e.type==pygame.MOUSEBUTTONDOWN and e.button==1:
            self.active = self.rect.collidepoint(e.pos)
        if self.active and e.type==pygame.KEYDOWN:
            if e.key==pygame.K_BACKSPACE: self.text=self.text[:-1]
            elif e.key==pygame.K_RETURN: self.active=False
            elif e.unicode and e.key!=pygame.K_TAB: self.text+=e.unicode

class SimpleEditor:
    def __init__(self):
        self.tracks = [Track("Track 1")]
        self.selected_track = 0
        self.selected_clip = None
        self.px_per_sec = 80.0  # zoom
        self.timeline_origin_x = 180
        self.track_h = 80
        self.ruler_h = 32
        self.top_bar_h = 60

        self.playhead = 0.0
        self.playing = False
        self.play_started_ms = None
        self.play_end_sec = None

        # Drag state
        self.drag_mode = None  # 'move', 'trim_l', 'trim_r', 'scrub'
        self.drag_offset_time = 0.0
        self.drag_clip_ref = None

        # UI controls
        self.btn_play = Button(10,10,60,30,"Play", self.play)
        self.btn_stop = Button(80,10,60,30,"Stop", self.stop)
        self.btn_save = Button(150,10,100,30,"Save WAV…", self.saveas)
        self.btn_add_track = Button(260,10,90,30,"+ Track", self.add_track)
        self.btn_import = Button(355,10,120,30,"Import Clip", self.import_clip)
        self.btn_delete = Button(480,10,90,30,"Delete", self.delete_selected)
        self.btn_split = Button(575,10,110,30,"Split @ Play", self.split_at_playhead)
        self.btn_zoom_in = Button(690,10,60,30,"+",
                                  lambda: self.set_zoom(self.px_per_sec*1.25))
        self.btn_zoom_out= Button(755,10,60,30,"-",
                                  lambda: self.set_zoom(self.px_per_sec/1.25))

        self.sld_gain = Slider(10, self.top_bar_h+5, 150, "Clip Gain", 0.0, 2.0, 1.0)
        self.sld_speed= Slider(10, self.top_bar_h+35, 150, "Clip Speed", 0.5, 1.5, 1.0)

        self.status = "Ready." ; self.status_col = COL_TXT
        self.mix_sound = None
        self.last_render = None  # (audio mono, start_sec)

    def set_status(self,msg,ok=True):
        self.status = msg; self.status_col = COL_OK if ok else COL_ERR

    def time_to_x(self, t_sec):
        return int(self.timeline_origin_x + t_sec * self.px_per_sec)

    def x_to_time(self, x):
        return max(0.0, (x - self.timeline_origin_x) / self.px_per_sec)

    def track_y(self, idx):
        return self.top_bar_h + self.ruler_h + idx*self.track_h

    def add_track(self):
        self.tracks.append(Track(f"Track {len(self.tracks)+0}"))
        self.set_status("Track added.")

    def import_clip(self):
        path = choose_open_file("Choose audio")
        if not path: self.set_status("Import cancelled.", ok=True); return
        try:
            y,_ = load_audio(path)
            name = os.path.basename(path)
            c = Clip(y, name=name, start=self.playhead, speed=1.0, gain=1.0)
            c.track_index = self.selected_track
            self.tracks[self.selected_track].clips.append(c)
            self.select_clip(c)
            self.set_status(f"Imported {name}")
        except Exception as e:
            self.set_status(f"Import failed: {e}", ok=False)

    def select_clip(self, clip):
        if self.selected_clip is not None:
            self.selected_clip.selected = False
        self.selected_clip = clip
        if clip is not None:
            clip.selected = True
            # sync sliders
            self.sld_gain.val = clip.gain
            self.sld_speed.val = clip.speed

    def delete_selected(self):
        c = self.selected_clip
        if not c: return
        tr = self.tracks[c.track_index]
        if c in tr.clips:
            tr.clips.remove(c)
        self.select_clip(None)
        self.set_status("Clip deleted.")

    def split_at_playhead(self):
        c = self.selected_clip
        if not c: return
        rc = c.cut_at(self.playhead)
        if rc is not None:
            rc.track_index = c.track_index
            self.tracks[c.track_index].clips.append(rc)
            self.set_status("Clip split.")
        else:
            self.set_status("Playhead not inside clip.", ok=False)

    # ===== Rendering / playback =====
    def project_length(self):
        end = 0.0
        for ti, tr in enumerate(self.tracks):
            for c in tr.clips:
                l,r = c.bounds()
                end = max(end, r)
        return end

    def render_mix(self, start_sec=0.0):
        # Figure end time
        end_sec = self.project_length()
        if end_sec <= start_sec + 1e-6:
            return np.zeros(1, dtype=np.float32), start_sec
        n0 = int(start_sec * SR)
        n_total = int((end_sec - start_sec) * SR)
        mix = np.zeros(n_total, dtype=np.float32)

        # Solo/mute logic
        any_solo = any(tr.solo for tr in self.tracks)

        for tr in self.tracks:
            if any_solo and not tr.solo: 
                continue
            if tr.mute: 
                continue
            t_gain = tr.volume
            for c in tr.clips:
                # Compute overlap between clip and render window
                left, right = c.bounds()
                if right <= start_sec or left >= end_sec:
                    continue
                # Source segment
                seg_in = c.in_pos
                seg_out = c.out_pos
                seg = c.audio[int(seg_in*SR): int(seg_out*SR)]
                if seg.size == 0: 
                    continue
                # Stretch by speed
                stretched = time_stretch(seg, c.speed)
                # Determine where to place in mix
                clip_len = len(stretched) / SR
                clip_start = c.start
                place_start = max(0.0, clip_start - start_sec)
                i0 = int(place_start * SR)
                # If clip truncated at left
                if clip_start < start_sec:
                    cut = int((start_sec - clip_start) * SR)
                    if cut < len(stretched):
                        stretched = stretched[cut:]
                    else:
                        continue
                # Truncate at project end
                if i0 >= n_total:
                    continue
                max_take = n_total - i0
                take = min(max_take, len(stretched))
                if take > 0:
                    mix[i0:i0+take] += stretched[:take] * (c.gain * t_gain)
        # Hard clip
        mix = np.clip(mix, -1.0, 1.0)
        return mix, start_sec

    def play(self):
        # Render from playhead to end
        audio, start = self.render_mix(self.playhead)
        if audio.size <= 1:
            self.set_status("Nothing to play.", ok=False); return
        st = to_int16_stereo(to_stereo(audio))
        samples = np.transpose(st)
        self.mix_sound = pygame.sndarray.make_sound(samples.copy())
        self.mix_sound.play()
        self.playing = True
        self.play_started_ms = pygame.time.get_ticks()
        self.play_end_sec = self.project_length()
        self.set_status("Playing…")

    def stop(self):
        pygame.mixer.stop()
        self.playing = False
        self.play_started_ms = None
        self.set_status("Stopped.")

    def saveas(self):
        # Render full
        audio, _ = self.render_mix(0.0)
        if audio.size <= 1:
            self.set_status("Nothing to save.", ok=False); return
        default = "mix.wav"
        path = choose_save_file("Save Mix As", default_name=default)
        if not path: self.set_status("Save cancelled.", ok=True); return
        try:
            st = to_int16_stereo(to_stereo(audio))
            with wave.open(path, 'wb') as wf:
                wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
                wf.writeframes(st.T.astype('<i2').tobytes())
            self.set_status(f"Saved: {path}")
        except Exception as e:
            self.set_status(f"Save failed: {e}", ok=False)

    # ===== Mouse / keyboard =====
    def pos_to_clip(self, x, y):
        # Determine which track row
        if y < self.top_bar_h + self.ruler_h:
            return None, None
        rel_y = y - (self.top_bar_h + self.ruler_h)
        row = int(rel_y // self.track_h)
        if row < 0 or row >= len(self.tracks):
            return None, None
        t = self.tracks[row]
        # Check clips
        for c in sorted(t.clips, key=lambda k: k.start, reverse=True):
            # Build rect like draw would
            x1 = self.time_to_x(c.start)
            x2 = self.time_to_x(c.start + max(0.05, c.eff_len_sec))
            r = pygame.Rect(x1, self.track_y(row)+6, x2-x1, self.track_h-12)
            # fuzzy handles
            if r.collidepoint(x,y):
                return c, r
        return None, None

    def on_mousedown(self, pos, button):
        x,y = pos
        # Click ruler to set playhead or scrub
        if self.top_bar_h <= y <= self.top_bar_h + self.ruler_h:
            self.playhead = self.x_to_time(x)
            self.drag_mode = 'scrub'
            return
        # Clips
        c, rect = self.pos_to_clip(x,y)
        if c is not None:
            self.selected_track = c.track_index
            self.select_clip(c)
            # edge handles
            handle = 8
            if abs(x - rect.left) <= handle:
                self.drag_mode = 'trim_l'
            elif abs(x - rect.right) <= handle:
                self.drag_mode = 'trim_r'
            else:
                self.drag_mode = 'move'
                self.drag_offset_time = self.x_to_time(x) - c.start
            self.drag_clip_ref = c
            return
        # Empty area click selects track
        row = int((y - (self.top_bar_h + self.ruler_h)) // self.track_h)
        if 0 <= row < len(self.tracks):
            self.selected_track = row
            self.select_clip(None)

    def on_mouseup(self, pos, button):
        self.drag_mode = None
        self.drag_clip_ref = None

    def on_mousemove(self, pos, buttons):
        x,y = pos
        if self.drag_mode == 'scrub' and buttons[0]:
            self.playhead = self.x_to_time(x)
            return
        c = self.drag_clip_ref
        if c is None: 
            return
        if self.drag_mode == 'move' and buttons[0]:
            new_start = max(0.0, self.x_to_time(x) - self.drag_offset_time)
            c.start = new_start
        elif self.drag_mode == 'trim_l' and buttons[0]:
            # compute new left time
            new_left_t = max(0.0, self.x_to_time(x))
            dt = new_left_t - c.start
            if dt != 0.0:
                # move start by dt, advance in_pos by dt*speed
                src_advance = dt * c.speed
                new_in = min(c.out_pos-0.01, max(0.0, c.in_pos + src_advance))
                # compute actual advancement applied
                applied = new_in - c.in_pos
                c.in_pos = new_in
                c.start = c.start + (applied / max(c.speed,1e-6))
        elif self.drag_mode == 'trim_r' and buttons[0]:
            new_right_t = max(c.start + 0.01, self.x_to_time(x))
            eff_len = new_right_t - c.start
            c.out_pos = c.in_pos + eff_len * c.speed

    def handle_keys(self, e):
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_SPACE:
                if self.playing: self.stop()
                else: self.play()
            elif e.key == pygame.K_s and (e.mod & pygame.KMOD_META or e.mod & pygame.KMOD_CTRL):
                self.saveas()
            elif e.key == pygame.K_DELETE or e.key == pygame.K_BACKSPACE:
                self.delete_selected()

    # ===== Draw =====
    def draw_ruler(self, surf):
        y0 = self.top_bar_h
        pygame.draw.rect(surf, COL_PANEL, pygame.Rect(0,y0, WIDTH, self.ruler_h))
        # seconds grid
        max_sec = max(10, int((WIDTH - self.timeline_origin_x) / self.px_per_sec) + 5)
        for s in range(max_sec):
            x = self.time_to_x(s)
            pygame.draw.line(surf, COL_GRID, (x, y0), (x, y0 + self.ruler_h))
            if s % 1 == 0:
                t = FONT_SM.render(str(s), True, COL_MUTED)
                surf.blit(t, (x+2, y0+8))
        # playhead
        px = self.time_to_x(self.playhead)
        pygame.draw.line(surf, (250, 90, 90), (px, y0), (px, HEIGHT), 2)

    def draw_tracks(self, surf):
        # track headers
        for i, tr in enumerate(self.tracks):
            y = self.track_y(i)
            pygame.draw.rect(surf, COL_PANEL, pygame.Rect(0,y, self.timeline_origin_x-2, self.track_h))
            pygame.draw.rect(surf, (60,60,80), pygame.Rect(0,y, self.timeline_origin_x-2, self.track_h), 2)
            name = f"{tr.name}  {'[M]' if tr.mute else ''}{'[S]' if tr.solo else ''}"
            surf.blit(FONT_MD.render(name, True, COL_TXT),(10, y+10))
        # clips
        area = pygame.Rect(self.timeline_origin_x, self.top_bar_h + self.ruler_h, WIDTH - self.timeline_origin_x, HEIGHT)
        pygame.draw.rect(surf, (28,28,38), area)
        for i, tr in enumerate(self.tracks):
            y = self.track_y(i)
            # row line
            pygame.draw.line(surf, (50,50,64), (self.timeline_origin_x, y), (WIDTH, y))
            for c in tr.clips:
                x1 = self.time_to_x(c.start)
                w = max(10, int(max(0.02, c.eff_len_sec) * self.px_per_sec))
                rect = pygame.Rect(x1, y+6, w, self.track_h-12)
                c.rect = rect
                col = COL_CLIP_SEL if c.selected else COL_CLIP
                pygame.draw.rect(surf, col, rect, border_radius=6)
                pygame.draw.rect(surf, (col[0]//2, col[1]//2, col[2]//2), rect, 2, border_radius=6)
                # handles
                pygame.draw.rect(surf, (240,240,240), pygame.Rect(rect.left-2, rect.top, 4, rect.h), 0)
                pygame.draw.rect(surf, (240,240,240), pygame.Rect(rect.right-2, rect.top, 4, rect.h), 0)
                # label
                label = f"{c.name}  x{c.speed:.2f}  g{c.gain:.2f}"
                surf.blit(FONT_SM.render(label, True, (15,15,18)), (rect.x+6, rect.y+4))

    def draw_topbar(self, surf):
        pygame.draw.rect(surf, COL_PANEL, pygame.Rect(0,0, WIDTH, self.top_bar_h))
        for b in [self.btn_play, self.btn_stop, self.btn_save, self.btn_add_track,
                  self.btn_import, self.btn_delete, self.btn_split, self.btn_zoom_in, self.btn_zoom_out]:
            b.draw(surf)
        # status
        surf.blit(FONT_SM.render(self.status, True, self.status_col),(860, 18))

    def set_zoom(self, new_pps):
        self.px_per_sec = max(20.0, min(400.0, float(new_pps)))

    def update_selected_sliders(self):
        c = self.selected_clip
        if c is None: return
        c.gain = self.sld_gain.val
        c.speed = self.sld_speed.val

    def run(self):
        clock = pygame.time.Clock()
        running = True
        # allow drop files
        try:
            pygame.event.set_allowed([pygame.QUIT, pygame.DROPFILE, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION, pygame.KEYDOWN])
        except Exception:
            pass

        while running:
            dt = clock.tick(60)
            mouse = pygame.mouse.get_pos()
            buttons = pygame.mouse.get_pressed()

            # update hover states
            for b in [self.btn_play, self.btn_stop, self.btn_save, self.btn_add_track,
                      self.btn_import, self.btn_delete, self.btn_split, self.btn_zoom_in, self.btn_zoom_out]:
                b.update(mouse)

            self.sld_gain.update(mouse, buttons)
            self.sld_speed.update(mouse, buttons)
            self.update_selected_sliders()

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN or e.type == pygame.KEYUP:
                    self.handle_keys(e)
                elif e.type == pygame.DROPFILE:
                    path = e.file
                    try:
                        y,_ = load_audio(path)
                        name = os.path.basename(path)
                        c = Clip(y, name=name, start=self.playhead, speed=1.0, gain=1.0)
                        c.track_index = self.selected_track
                        self.tracks[self.selected_track].clips.append(c)
                        self.select_clip(c)
                        self.set_status(f"Imported (drop) {name}")
                    except Exception as ex:
                        self.set_status(f"Drop import failed: {ex}", ok=False)
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button==1:
                    self.on_mousedown(e.pos, e.button)
                    # buttons
                    for b in [self.btn_play, self.btn_stop, self.btn_save, self.btn_add_track,
                              self.btn_import, self.btn_delete, self.btn_split, self.btn_zoom_in, self.btn_zoom_out]:
                        b.handle(e)
                elif e.type == pygame.MOUSEBUTTONUP and e.button==1:
                    self.on_mouseup(e.pos, e.button)
                elif e.type == pygame.MOUSEMOTION:
                    self.on_mousemove(e.pos, buttons)

            # update playhead while playing
            if self.playing and self.play_started_ms is not None:
                elapsed = (pygame.time.get_ticks() - self.play_started_ms) / 1000.0
                self.playhead = min(self.play_end_sec, self.playhead + elapsed)  # naive increment
                self.play_started_ms = pygame.time.get_ticks()
                if self.playhead >= self.play_end_sec - 1e-3:
                    self.stop()

            # draw
            screen.fill(COL_BG)
            self.draw_topbar(screen)
            self.draw_ruler(screen)
            # controls panel on left
            pygame.draw.rect(screen, COL_PANEL, pygame.Rect(0, self.top_bar_h, 170, self.ruler_h + len(self.tracks)*self.track_h))
            self.sld_gain.draw(screen); self.sld_speed.draw(screen)
            self.draw_tracks(screen)

            pygame.display.flip()
        pygame.quit()

def main():
    app = SimpleEditor()
    # Optional CLI import
    if len(sys.argv)>1 and os.path.isfile(sys.argv[1]):
        try:
            y,_ = load_audio(sys.argv[1])
            name = os.path.basename(sys.argv[1])
            c = Clip(y, name=name, start=0.0, speed=1.0, gain=1.0)
            c.track_index = 0
            app.tracks[0].clips.append(c)
            app.select_clip(c)
            app.set_status(f"Loaded {name} from CLI.")
        except Exception as e:
            app.set_status(f"CLI load failed: {e}", ok=False)
    app.run()

if __name__ == "__main__":
    main()
