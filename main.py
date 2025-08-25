import os
import sys
import math
import wave
import time
import warnings
import subprocess
from typing import List, Tuple, Optional

import numpy as np
import pygame

# ---------- Optional loader / stretch ----------
try:
    import librosa
    import soundfile as sf  # noqa: F401
    LIBROSA_AVAILABLE = True
except Exception:
    LIBROSA_AVAILABLE = False

SR = 44100
warnings.filterwarnings("ignore", message=r".*__audioread_load.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=r"PySoundFile failed.*")

# ---------- macOS-safe file dialogs ----------
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
            fp = filedialog.askopenfilename(
                title=title,
                filetypes=[("Audio", "*.wav *.mp3 *.flac *.aiff *.aif *.ogg"), ("All files", "*.*")]
            )
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
            fp = filedialog.asksaveasfilename(
                title=title, defaultextension=".wav", filetypes=[("WAV", "*.wav")], initialfile=default_name
            )
            root.destroy()
            return fp if fp else None
        except Exception:
            return None
    return None


def choose_folder(title="Choose Folder"):
    if sys.platform == "darwin":
        try:
            osa = f'POSIX path of (choose folder with prompt "{title}")'
            res = subprocess.run(["osascript", "-e", osa], capture_output=True, text=True)
            out = res.stdout.strip()
            return out if res.returncode == 0 and out else None
        except Exception:
            return None
    if TK_AVAILABLE:
        try:
            root = tk.Tk(); root.withdraw(); root.update()
            fp = filedialog.askdirectory(title=title)
            root.destroy()
            return fp if fp else None
        except Exception:
            return None
    return None


# ---------- Audio helpers ----------
def load_audio(path: str, target_sr: int = SR) -> Tuple[np.ndarray, int]:
    """Return mono float32 [-1,1], sr."""
    if not os.path.exists(path):
        return np.array([], dtype=np.float32), target_sr
        
    if LIBROSA_AVAILABLE:
        try:
            y, _ = librosa.load(path, sr=target_sr, mono=True)
            y = y.astype(np.float32)
            m = float(np.max(np.abs(y))) if y.size else 0.0
            if m > 0:
                y = y / m
            return y, target_sr
        except Exception as e:
            print(f"Librosa load failed: {e}")
            # Fall through to wave method
    
    try:
        with wave.open(path, 'rb') as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            ch = wf.getnchannels()
            data = wf.readframes(n)
            dtype = np.int16 if wf.getsampwidth() == 2 else np.int8
            arr = np.frombuffer(data, dtype=dtype).astype(np.float32)
            
            # Normalize based on bit depth
            if dtype == np.int16:
                arr /= 32768.0
            else:
                arr /= 128.0
                
            if ch > 1:
                arr = arr.reshape(-1, ch).mean(axis=1)
            if sr != target_sr and arr.size:
                t_old = np.linspace(0, len(arr)/sr, len(arr), endpoint=False)
                t_new = np.linspace(0, len(arr)/sr, int(len(arr)*target_sr/sr), endpoint=False)
                arr = np.interp(t_new, t_old, arr).astype(np.float32)
                sr = target_sr
            m = float(np.max(np.abs(arr))) if arr.size else 0.0
            if m > 0:
                arr = arr / m
            return arr, sr
    except Exception as e:
        print(f"Wave load failed: {e}")
        return np.array([], dtype=np.float32), target_sr


def time_stretch(y: np.ndarray, speed: float) -> np.ndarray:
    """speed>1 → faster (shorter). If no librosa, naive resample (pitch changes)."""
    speed = max(0.25, min(2.0, float(speed)))
    if not y.size or abs(speed - 1.0) < 1e-6:
        return y.astype(np.float32)
        
    if LIBROSA_AVAILABLE:
        try:
            return librosa.effects.time_stretch(y, rate=speed).astype(np.float32)
        except Exception:
            # Fallback to interpolation if librosa fails
            pass
            
    n_new = max(1, int(len(y) / speed))
    t_old = np.linspace(0, 1, len(y), endpoint=False)
    t_new = np.linspace(0, 1, n_new, endpoint=False)
    return np.interp(t_new, t_old, y).astype(np.float32)


def to_stereo(mono: np.ndarray) -> np.ndarray:
    """Return (2, n) float32."""
    if mono.size == 0:
        return np.zeros((2, 0), dtype=np.float32)
    return np.vstack([mono, mono]).astype(np.float32, copy=False)


def to_int16_stereo(st: np.ndarray) -> np.ndarray:
    """(2, n) float32 -> (n, 2) int16 contiguous."""
    if st.size == 0:
        return np.zeros((0, 2), dtype=np.int16)
        
    st = np.clip(st, -1.0, 1.0)
    arr = (st * 32767.0).astype(np.int16, copy=False).T
    return np.ascontiguousarray(arr)  # (n,2) C-contiguous


def apply_fades(x: np.ndarray, fin_sec: float, fout_sec: float, sr: int = SR) -> np.ndarray:
    """In-place linear fades."""
    n = len(x)
    if n == 0:
        return x
    if fin_sec > 0:
        m = min(n, int(fin_sec * sr))
        if m > 0:
            x[:m] *= np.linspace(0, 1, m, dtype=np.float32)
    if fout_sec > 0:
        m = min(n, int(fout_sec * sr))
        if m > 0:
            x[-m:] *= np.linspace(1, 0, m, dtype=np.float32)
    return x


# ---------- Data models ----------
class Clip:
    def __init__(self, audio: np.ndarray, name: str, start=0.0, speed=1.0, gain=1.0,
                 in_pos=0.0, out_pos=None, fade_in=0.02, fade_out=0.02):
        self.audio = audio.astype(np.float32)
        self.name = name
        self.start = float(start)    # timeline start (sec)
        self.speed = float(speed)    # 1.0 normal; 0.5 slower; 1.5 faster
        self.gain = float(gain)
        self.in_pos = float(in_pos)  # seconds into source
        self.out_pos = float(out_pos) if out_pos is not None else (len(audio)/SR)
        self.fade_in = float(max(0.0, fade_in))
        self.fade_out = float(max(0.0, fade_out))
        self.track_index = 0
        self.selected = False
        self.rect: Optional[pygame.Rect] = None  # UI cache

    @property
    def eff_len_sec(self) -> float:
        return max(0.0, (self.out_pos - self.in_pos) / max(self.speed, 1e-6))

    def bounds(self) -> Tuple[float, float]:
        return self.start, self.start + self.eff_len_sec

    def cut_at(self, t_sec: float) -> Optional["Clip"]:
        """Split at timeline time t_sec; return right piece or None if outside."""
        left, right = self.bounds()
        if t_sec <= left + 1e-6 or t_sec >= right - 1e-6:
            return None
        dt = t_sec - self.start
        consumed = dt * self.speed
        split_src = self.in_pos + consumed
        right_clip = Clip(self.audio, name=self.name + " (split)", start=t_sec,
                          speed=self.speed, gain=self.gain, in_pos=split_src, out_pos=self.out_pos,
                          fade_in=self.fade_in, fade_out=self.fade_out)
        self.out_pos = split_src
        return right_clip


class Track:
    def __init__(self, name: str):
        self.name = name
        self.volume = 1.0
        self.mute = False
        self.solo = False
        self.clips: List[Clip] = []
        # runtime
        self.meter_level = 0.0       # 0..1 (post-fader)
        self._fader_rect: Optional[pygame.Rect] = None
        self._mute_rect: Optional[pygame.Rect] = None
        self._solo_rect: Optional[pygame.Rect] = None


# ---------- UI / mixer ----------
# Initialize pygame display first, then mixer
pygame.init()
# Ensure sndarray uses NumPy
pygame.sndarray.use_arraytype("numpy")

WIDTH, HEIGHT = 1280, 860
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Simple Audio Editor — Fixed Version")

# Try to initialize mixer with different parameters
mixer_initialized = False
for buffer_size in [1024, 2048, 4096, 8192]:
    try:
        pygame.mixer.init(frequency=SR, size=-16, channels=2, buffer=buffer_size)
        pygame.mixer.set_num_channels(64)
        mixer_initialized = True
        print(f"Mixer initialized with buffer size: {buffer_size}")
        break
    except pygame.error as e:
        print(f"Failed to initialize mixer with buffer {buffer_size}: {e}")

if not mixer_initialized:
    try:
        pygame.mixer.init()
        print("Mixer initialized with default settings")
    except pygame.error as e:
        print(f"Failed to initialize mixer: {e}")
        sys.exit(1)

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

pygame.font.init()
FONT_MD = pygame.font.SysFont("Arial", 16, bold=True)
FONT_SM = pygame.font.SysFont("Arial", 12)


class Button:
    def __init__(self, x, y, w, h, label, fn, toggle=False, state=False):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.fn = fn
        self.toggle = toggle
        self.state = state
        self.hover = False

    def draw(self, surf):
        base = COL_ACC2 if (self.toggle and self.state) else COL_ACC
        col = base if not self.hover else (min(base[0]+40, 255), min(base[1]+40, 255), min(base[2]+40, 255))
        pygame.draw.rect(surf, col, self.rect, border_radius=6)
        pygame.draw.rect(surf, (col[0]//2, col[1]//2, col[2]//2), self.rect, 2, border_radius=6)
        t = FONT_SM.render(self.label, True, COL_TXT)
        surf.blit(t, t.get_rect(center=self.rect.center))

    def update(self, mouse):
        self.hover = self.rect.collidepoint(mouse)

    def handle_click(self):
        if self.toggle:
            self.state = not self.state
        if self.fn:
            self.fn()


class Slider:
    def __init__(self, x, y, w, label, minv, maxv, val):
        self.rect = pygame.Rect(x, y, w, 20)
        self.label = label
        self.min = minv
        self.max = maxv
        self.val = val
        self.drag = False

    def draw(self, surf):
        surf.blit(FONT_SM.render(f"{self.label}: {self.val:.2f}", True, COL_TXT), (self.rect.x, self.rect.y - 16))
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


# ---------- Editor ----------
class SimpleEditor:
    def __init__(self):
        # layout
        self.timeline_origin_x = 230
        self.track_h = 100
        self.ruler_h = 36
        self.top_bar_h = 60
        self.bottom_bar_h = 16
        self.right_bar_w = 12

        # view / scroll
        self.px_per_sec = 100.0
        self.h_off_sec = 0.0
        self.v_off_px = 0
        self._drag_scroll_h = False
        self._drag_scroll_v = False

        # grid / snapping
        self.bpm = 120.0
        self.subdiv = 4
        self.snap = True

        # model
        self.tracks: List[Track] = []
        self._create_track()
        self._create_track()
        self.selected_track = 0
        self.selected_clip: Optional[Clip] = None

        # playback
        self.playhead = 0.0
        self.playing = False
        self.play_start_clock: Optional[float] = None
        self.play_start_pos: float = 0.0
        self.play_end_sec: Optional[float] = None
        self._stems: Optional[List[np.ndarray]] = None
        self.mix_sound: Optional[pygame.mixer.Sound] = None
        self.play_channel: Optional[pygame.mixer.Channel] = None

        # drag state
        self.drag_mode: Optional[str] = None
        self.drag_offset_time = 0.0
        self.drag_clip_ref: Optional[Clip] = None
        self.drag_fader_track: Optional[int] = None
        self.drag_last_mouse = (0, 0)

        # UI
        x = 10
        self.btn_play = Button(x, 10, 60, 30, "Play", self.play); x += 70
        self.btn_stop = Button(x, 10, 60, 30, "Stop", self.stop); x += 70
        self.btn_tone = Button(x, 10, 70, 30, "Test Tone", self.test_tone); x += 80
        self.btn_save = Button(x, 10, 110, 30, "Save Mix…", self.saveas); x += 120
        self.btn_save_stems = Button(x, 10, 130, 30, "Save Stems…", self.save_stems); x += 140
        self.btn_add_track = Button(x, 10, 90, 30, "+ Track", self.add_track); x += 100
        self.btn_import = Button(x, 10, 120, 30, "Import Clip", self.import_clip); x += 130
        self.btn_delete = Button(x, 10, 90, 30, "Delete", self.delete_selected); x += 100
        self.btn_split = Button(x, 10, 120, 30, "Split @ Play", self.split_at_playhead); x += 130
        self.btn_zoom_in = Button(x, 10, 48, 30, "+", lambda: self.set_zoom(self.px_per_sec * 1.25)); x += 58
        self.btn_zoom_out = Button(x, 10, 48, 30, "-", lambda: self.set_zoom(self.px_per_sec / 1.25)); x += 58
        self.btn_snap = Button(x, 10, 70, 30, "Snap", self.toggle_snap, toggle=True, state=self.snap); x += 80

        self.sld_gain = Slider(10, self.top_bar_h + 6, 190, "Clip Gain", 0.0, 2.0, 1.0)
        self.sld_speed = Slider(10, self.top_bar_h + 36, 190, "Clip Speed", 0.5, 1.5, 1.0)
        self.sld_bpm = Slider(10, self.top_bar_h + 66, 190, "BPM", 60.0, 200.0, self.bpm)

        self.status = "Ready."
        self.status_col = COL_TXT

    # ---------- track helpers ----------
    def _create_track(self):
        idx = len(self.tracks) + 1
        self.tracks.append(Track(f"Track {idx}"))

    def add_track(self):
        self._create_track()
        self.selected_track = len(self.tracks) - 1
        self.set_status(f"Added Track {self.selected_track + 1}")

    def _ensure_track_for_y(self, y_px: int) -> int:
        """Return a valid track index for a mouse y; auto-create if below last lane."""
        top = self.top_bar_h + self.ruler_h
        if y_px < top:
            return self.selected_track
        lane = (y_px + self.v_off_px - top) // self.track_h
        if lane < 0:
            return self.selected_track
        if lane >= len(self.tracks):
            while len(self.tracks) <= lane:
                self._create_track()
            self.set_status(f"Auto-created Track {int(lane)+1}")
        return int(lane)

    # ---------- misc helpers ----------
    def set_status(self, msg, ok=True):
        self.status = msg
        self.status_col = COL_OK if ok else COL_ERR
        print(msg)

    def visible_secs(self) -> float:
        return max(0.5, (WIDTH - self.timeline_origin_x - self.right_bar_w) / self.px_per_sec)

    def total_track_pixels(self) -> int:
        return len(self.tracks) * self.track_h

    def set_zoom(self, new_pps: float, anchor_time: Optional[float] = None):
        if anchor_time is None:
            anchor_time = self.h_off_sec + self.visible_secs() / 2
        new_pps = max(30.0, min(640.0, float(new_pps)))
        self.px_per_sec = new_pps
        self.h_off_sec = max(0.0, anchor_time - self.visible_secs() / 2)

    def maybe_snap(self, t: float) -> float:
        if not self.snap:
            return max(0.0, t)
        beat = 60.0 / max(1e-6, self.bpm)
        step = beat / max(1, self.subdiv)
        return max(0.0, round(t / step) * step)

    def time_to_x(self, t_sec: float) -> int:
        return int(self.timeline_origin_x + (t_sec - self.h_off_sec) * self.px_per_sec)

    def x_to_time(self, x: int) -> float:
        return self.h_off_sec + (x - self.timeline_origin_x) / self.px_per_sec

    def track_y(self, idx: int) -> int:
        return self.top_bar_h + self.ruler_h + idx * self.track_h - self.v_off_px

    def project_length(self) -> float:
        end = 0.0
        for tr in self.tracks:
            for c in tr.clips:
                _, right = c.bounds()
                end = max(end, right)
        return end

    def toggle_snap(self):
        self.snap = not self.snap
        self.btn_snap.state = self.snap
        self.set_status(f"Snap {'ON' if self.snap else 'OFF'}")

    # ---------- selection ----------
    def select_clip(self, clip: Optional[Clip]):
        if self.selected_clip is not None:
            self.selected_clip.selected = False
        self.selected_clip = clip
        if clip is not None:
            clip.selected = True
            if self.drag_mode is None:
                self.sld_gain.val = clip.gain
                self.sld_speed.val = clip.speed

    # ---------- import / delete / split ----------
    def import_clip(self):
        path = choose_open_file("Choose audio")
        if not path:
            self.set_status("Import cancelled.", ok=True)
            return
        mx, my = pygame.mouse.get_pos()
        self._import_at_screen_pos(path, mx, my)

    def _import_at_screen_pos(self, path: str, mx: int, my: int):
        try:
            y, _ = load_audio(path)
            if y.size == 0:
                self.set_status(f"Failed to load audio from {path}", ok=False)
                return
            name = os.path.basename(path)
            start_t = self.playhead
            ti = self._ensure_track_for_y(my)
            c = Clip(y, name=name, start=start_t, speed=1.0, gain=1.0)
            c.track_index = ti
            self.tracks[ti].clips.append(c)
            self.select_clip(c)
            self.invalidate_render()
            self.set_status(f"Imported {name} → Track {ti+1}")
        except Exception as e:
            self.set_status(f"Import failed: {e}", ok=False)

    def delete_selected(self):
        c = self.selected_clip
        if not c:
            return
        tr = self.tracks[c.track_index]
        if c in tr.clips:
            tr.clips.remove(c)
        self.select_clip(None)
        self.invalidate_render()
        self.set_status("Clip deleted.")

    def split_at_playhead(self):
        c = self.selected_clip
        if not c:
            return
        rc = c.cut_at(self.playhead)
        if rc is not None:
            rc.track_index = c.track_index
            self.tracks[c.track_index].clips.append(rc)
            self.invalidate_render()
            self.set_status("Clip split.")
        else:
            self.set_status("Playhead not inside clip.", ok=False)

    # ---------- rendering / stems ----------
    def render_mix_with_stems(self, start_sec: float = 0.0) -> Tuple[np.ndarray, float, List[np.ndarray]]:
        end_sec = max(self.project_length(), start_sec + 0.01)
        n_total = int((end_sec - start_sec) * SR)
        stems = [np.zeros(n_total, dtype=np.float32) for _ in self.tracks]

        any_solo = any(tr.solo for tr in self.tracks)

        for ti, tr in enumerate(self.tracks):
            if any_solo and not tr.solo:
                continue
            if tr.mute:
                continue
            stem = stems[ti]
            for c in tr.clips:
                left, right = c.bounds()
                if right <= start_sec or left >= end_sec:
                    continue
                    
                # Get the audio segment
                start_sample = int(c.in_pos * SR)
                end_sample = int(c.out_pos * SR)
                if start_sample >= end_sample:
                    continue
                    
                seg = c.audio[start_sample:end_sample].copy()
                if seg.size == 0:
                    continue
                    
                # Apply time stretching
                stretched = time_stretch(seg, c.speed)
                if stretched.size == 0:
                    continue
                    
                # Apply fades
                apply_fades(stretched, c.fade_in, c.fade_out, SR)
                
                # Calculate placement
                clip_start = c.start
                place_start = max(0.0, clip_start - start_sec)
                i0 = int(place_start * SR)
                
                # Handle case where clip starts before our render start
                if clip_start < start_sec:
                    cut = int((start_sec - clip_start) * SR)
                    if cut < len(stretched):
                        stretched = stretched[cut:]
                    else:
                        stretched = np.zeros(0, dtype=np.float32)
                
                if stretched.size == 0 or i0 >= n_total:
                    continue
                    
                take = min(n_total - i0, len(stretched))
                if take > 0:
                    stem[i0:i0+take] += stretched[:take] * c.gain
                    
            stems[ti] = stems[ti] * tr.volume

        mix = np.sum(stems, axis=0)

        # Headroom + gentle limiter + auto-gain to ~-1 dBFS
        if mix.size:
            peak = float(np.max(np.abs(mix)))
            if peak > 1.0:
                mix /= (peak + 1e-6)
            mix = np.tanh(mix * 1.2)
            peak2 = float(np.max(np.abs(mix)))
            target = 0.89  # about -1 dBFS
            if peak2 > 1e-6 and peak2 < target:
                mix *= (target / peak2)

        return mix, start_sec, stems

    def invalidate_render(self):
        self._stems = None
        self.mix_sound = None

    # ---------- play/stop ----------
    def _debug_print_buf(self, mono: np.ndarray, stereo_i16: np.ndarray, start_idx: int):
        rms = float(np.sqrt(np.mean(mono[start_idx:start_idx+min(SR, max(1, mono.size - start_idx))]**2))) if mono.size > start_idx else 0.0
        pk = float(np.max(np.abs(mono))) if mono.size else 0.0
        print(f"[PLAY] start_idx={start_idx} mono_len={mono.size} pk={pk:.3f} rms≈{rms:.3f} "
              f"stereo_i16.shape={stereo_i16.shape} dtype={stereo_i16.dtype}")

    def _play_array(self, mono: np.ndarray, start_idx: int):
        if mono.size == 0 or start_idx >= mono.size:
            self.set_status("Nothing to play at playhead.", ok=False)
            return False
            
        chunk = mono[start_idx:]
        if not chunk.size or float(np.max(np.abs(chunk))) < 1e-6:
            self.set_status("Buffer is silent at playhead.", ok=False)
            return False
            
        stereo = to_stereo(chunk)              # (2,n) float32
        stereo_i16 = to_int16_stereo(stereo)   # (n,2) int16 contiguous
        
        try:
            snd = pygame.sndarray.make_sound(stereo_i16)  # expects (n,2) int16
        except Exception as e:
            self.set_status(f"make_sound failed: {e}", ok=False)
            return False
            
        pygame.mixer.stop()
        try:
            ch = snd.play(loops=0, fade_ms=10)
            if ch is None:
                self.set_status("Channel play failed.", ok=False)
                return False
        except pygame.error as e:
            self.set_status(f"Play failed: {e}", ok=False)
            return False
            
        self.mix_sound = snd
        self.play_channel = ch
        return True

    def play(self):
        if self.playing:
            self.stop()
            return
            
        try:
            mix, start, stems = self.render_mix_with_stems(0.0)
            self._stems = stems
            i0 = int(self.playhead * SR)
            ok = self._play_array(mix, i0)
            if not ok:
                return
                
            self.play_start_pos = self.playhead
            self.play_start_clock = time.perf_counter()
            self.play_end_sec = len(mix) / SR
            self.playing = True
            self.set_status("Playing…", ok=True)
        except Exception as e:
            self.set_status(f"Play failed: {e}", ok=False)

    def stop(self):
        try:
            pygame.mixer.stop()
        except:
            pass
        self.playing = False
        self.play_channel = None
        self.play_start_clock = None
        self.set_status("Stopped.", ok=True)

    def test_tone(self):
        """1s 440 Hz beep to confirm output device/mixer are OK."""
        t = np.linspace(0, 1.0, SR, endpoint=False).astype(np.float32)
        beep = 0.3 * np.sin(2 * np.pi * 440.0 * t)
        stereo = to_stereo(beep)
        stereo_i16 = to_int16_stereo(stereo)
        try:
            snd = pygame.sndarray.make_sound(stereo_i16)
            pygame.mixer.stop()
            ch = snd.play()
            if ch is None:
                self.set_status("Tone play failed (no channel).", ok=False)
                return
            self.mix_sound = snd
            self.playing = True
            self.play_start_pos = 0.0
            self.play_start_clock = time.perf_counter()
            self.play_end_sec = 1.0
            self.set_status("Test tone playing (1s)…", ok=True)
        except Exception as e:
            self.set_status(f"Test tone failed: {e}", ok=False)

    # ---------- export ----------
    def saveas(self):
        path = choose_save_file("Save Mix As", "mix.wav")
        if not path:
            self.set_status("Save cancelled.", ok=True); return
        mix, start, _ = self.render_mix_with_stems(0.0)
        stereo = to_stereo(mix)
        try:
            with wave.open(path, 'wb') as wf:
                wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
                wf.writeframes(to_int16_stereo(stereo).tobytes())
            self.set_status(f"Saved mix → {path}")
        except Exception as e:
            self.set_status(f"Save failed: {e}", ok=False)

    def save_stems(self):
        folder = choose_folder("Choose folder for stems")
        if not folder:
            self.set_status("Stem export cancelled.", ok=True); return
        mix, start, stems = self.render_mix_with_stems(0.0)
        try:
            for i, (tr, stem) in enumerate(zip(self.tracks, stems)):
                st = to_stereo(stem)
                name = f"{i+1:02d}_{tr.name.replace(' ', '_')}.wav"
                out = os.path.join(folder, name)
                with wave.open(out, 'wb') as wf:
                    wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
                    wf.writeframes(to_int16_stereo(st).tobytes())
            self.set_status(f"Exported stems → {folder}")
        except Exception as e:
            self.set_status(f"Stem export failed: {e}", ok=False)

    # ---------- meters ----------
    def update_meters(self):
        if not self.playing or self._stems is None:
            for tr in self.tracks: tr.meter_level = 0.0
            return
            
        i = int(self.playhead * SR)
        win = int(0.05 * SR)  # 50ms
        for tr, stem in zip(self.tracks, self._stems):
            if i >= len(stem):
                tr.meter_level = 0.0; continue
            s = stem[i:i+win]
            tr.meter_level = float(min(1.0, max(0.0, np.sqrt(np.mean((s * tr.volume) ** 2)) * 2.0)))

    # ---------- drawing ----------
    def draw(self, surf):
        surf.fill(COL_BG)
        self.draw_topbar(surf)
        self.draw_ruler_and_grid(surf)
        self.draw_tracks(surf)
        self.draw_playhead(surf)
        self.draw_scrollbars(surf)
        txt = FONT_SM.render(self.status, True, self.status_col)
        surf.blit(txt, (10, HEIGHT - self.bottom_bar_h + 2))

    def draw_topbar(self, surf):
        pygame.draw.rect(surf, COL_PANEL, (0, 0, WIDTH, self.top_bar_h))
        mouse = pygame.mouse.get_pos()
        for b in [self.btn_play, self.btn_stop, self.btn_tone, self.btn_save, self.btn_save_stems,
                  self.btn_add_track, self.btn_import, self.btn_delete,
                  self.btn_split, self.btn_zoom_in, self.btn_zoom_out, self.btn_snap]:
            b.update(mouse); b.draw(surf)
        self.sld_gain.draw(surf); self.sld_speed.draw(surf); self.sld_bpm.draw(surf)

    def draw_ruler_and_grid(self, surf):
        y0 = self.top_bar_h
        pygame.draw.rect(surf, (28, 28, 38), (0, y0, WIDTH, self.ruler_h))
        left_t = self.h_off_sec
        right_t = self.h_off_sec + self.visible_secs()
        beat = 60.0 / max(1e-6, self.bpm)
        bar = beat * 4
        first_bar = math.floor(left_t / bar)
        t = first_bar * bar
        while t < right_t + bar:
            x = self.time_to_x(t)
            pygame.draw.line(surf, COL_GRID_BAR, (x, y0), (x, HEIGHT - self.bottom_bar_h), 2)
            for b in range(1, 4):
                tb = t + b * beat
                if left_t <= tb <= right_t:
                    xb = self.time_to_x(tb)
                    pygame.draw.line(surf, COL_GRID, (xb, y0 + self.ruler_h//2), (xb, HEIGHT - self.bottom_bar_h), 1)
            if x >= self.timeline_origin_x - 30:
                label = FONT_SM.render(str(int(t / bar)), True, COL_TXT)
                surf.blit(label, (x - 4, y0 + 4))
            t += bar
        pygame.draw.rect(surf, COL_PANEL, (0, self.top_bar_h + self.ruler_h, self.timeline_origin_x, HEIGHT - self.top_bar_h - self.ruler_h))

    def draw_tracks(self, surf):
        for i, tr in enumerate(self.tracks):
            y = self.track_y(i)
            if y + self.track_h < self.top_bar_h + self.ruler_h or y > HEIGHT - self.bottom_bar_h:
                continue
                
            # header
            hdr = pygame.Rect(0, y, self.timeline_origin_x, self.track_h - 4)
            pygame.draw.rect(surf, (30, 30, 40), hdr)
            pygame.draw.rect(surf, (55, 55, 70), hdr, 1)
            title_col = COL_TXT if i != self.selected_track else COL_ACC
            surf.blit(FONT_MD.render(tr.name, True, title_col), (10, y + 6))
            tr._mute_rect = pygame.Rect(12, y + 30, 24, 18)
            tr._solo_rect = pygame.Rect(42, y + 30, 24, 18)
            pygame.draw.rect(surf, COL_MUTE if tr.mute else (70, 70, 80), tr._mute_rect, border_radius=4)
            pygame.draw.rect(surf, COL_SOLO if tr.solo else (70, 70, 80), tr._solo_rect, border_radius=4)
            surf.blit(FONT_SM.render("M", True, COL_TXT), tr._mute_rect.move(7, 1))
            surf.blit(FONT_SM.render("S", True, COL_TXT), tr._solo_rect.move(7, 1))
            fx, fw = 80, self.timeline_origin_x - 100
            tr._fader_rect = pygame.Rect(fx, y + 30, fw, 6)
            pygame.draw.rect(surf, (60, 60, 80), tr._fader_rect, border_radius=3)
            knob_x = int(fx + tr.volume * (fw - 1))
            pygame.draw.circle(surf, COL_ACC2, (knob_x, y + 33), 7)
            # meter
            meter = pygame.Rect(self.timeline_origin_x - 20, y + 6, 8, self.track_h - 20)
            pygame.draw.rect(surf, COL_METER_BG, meter)
            m_h = int((self.track_h - 20) * max(0.0, min(1.0, tr.meter_level)))
            if m_h > 0:
                pygame.draw.rect(surf, COL_METER, (meter.x, meter.bottom - m_h, meter.w, m_h))
            # lane
            lane = pygame.Rect(self.timeline_origin_x, y, WIDTH - self.timeline_origin_x - self.right_bar_w, self.track_h - 4)
            col = (24, 24, 30) if i % 2 == 0 else (22, 22, 28)
            pygame.draw.rect(surf, col, lane)
            # clips
            for c in tr.clips:
                left, right = c.bounds()
                if right < self.h_off_sec or left > self.h_off_sec + self.visible_secs():
                    continue
                x1 = self.time_to_x(left)
                x2 = self.time_to_x(right)
                rect = pygame.Rect(x1, y + 6, max(32, x2 - x1), self.track_h - 16)
                c.rect = rect
                pygame.draw.rect(surf, COL_CLIP_SEL if c.selected else COL_CLIP, rect, border_radius=6)
                pygame.draw.rect(surf, (50, 50, 70), rect, 2, border_radius=6)
                label = FONT_SM.render(f"{c.name}  x{c.speed:.2f}  g{c.gain:.2f}", True, COL_TXT)
                surf.blit(label, (rect.x + 6, rect.y + 4))
                # fade handles
                fh = 10
                pts_l = [(rect.x, rect.y), (rect.x + fh, rect.y), (rect.x, rect.y + fh)]
                pts_r = [(rect.right, rect.y), (rect.right - fh, rect.y), (rect.right, rect.y + fh)]
                pygame.draw.polygon(surf, COL_FADE, pts_l)
                pygame.draw.polygon(surf, COL_FADE, pts_r)

    def draw_playhead(self, surf):
        x = self.time_to_x(self.playhead)
        pygame.draw.line(surf, (240, 80, 80), (x, self.top_bar_h), (x, HEIGHT - self.bottom_bar_h), 2)

    def draw_scrollbars(self, surf):
        # Horizontal
        y = HEIGHT - self.bottom_bar_h - 10
        pygame.draw.rect(surf, COL_SCROLL, (self.timeline_origin_x, y, WIDTH - self.timeline_origin_x - self.right_bar_w, 10))
        total = max(self.project_length(), self.visible_secs())
        vis = self.visible_secs()
        if total > 0:
            frac = vis / max(total, 1e-9)
            frac = max(0.05, min(1.0, frac))
            span_px = (WIDTH - self.timeline_origin_x - self.right_bar_w) * frac
            max_off = max(0.0, total - vis)
            off_frac = 0.0 if max_off <= 0 else (self.h_off_sec / max_off) * (1 - frac)
            knob_x = int(self.timeline_origin_x + off_frac * (WIDTH - self.timeline_origin_x - self.right_bar_w))
            self.h_scroll_rect = pygame.Rect(knob_x, y, int(span_px), 10)
            pygame.draw.rect(surf, COL_SCROLL_KNOB, self.h_scroll_rect)
        else:
            self.h_scroll_rect = pygame.Rect(self.timeline_origin_x, y, WIDTH - self.timeline_origin_x - self.right_bar_w, 10)

        # Vertical
        x = WIDTH - self.right_bar_w
        pygame.draw.rect(surf, COL_SCROLL, (x, self.top_bar_h + self.ruler_h, self.right_bar_w, HEIGHT - self.top_bar_h - self.ruler_h - self.bottom_bar_h))
        total_px = self.total_track_pixels()
        vis_px = HEIGHT - self.top_bar_h - self.ruler_h - self.bottom_bar_h
        if total_px > 0:
            frac = vis_px / max(total_px, 1)
            frac = max(0.05, min(1.0, frac))
            span = int((HEIGHT - self.top_bar_h - self.ruler_h - self.bottom_bar_h) * frac)
            max_off = max(0, total_px - vis_px)
            off_frac = 0.0 if max_off <= 0 else (self.v_off_px / max_off) * (1 - frac)
            knob_y = int(self.top_bar_h + self.ruler_h + off_frac * (HEIGHT - self.top_bar_h - self.ruler_h - self.bottom_bar_h))
            self.v_scroll_rect = pygame.Rect(x, knob_y, self.right_bar_w, span)
            pygame.draw.rect(surf, COL_SCROLL_KNOB, self.v_scroll_rect)
        else:
            self.v_scroll_rect = pygame.Rect(x, self.top_bar_h + self.ruler_h, self.right_bar_w, HEIGHT - self.top_bar_h - self.ruler_h - self.bottom_bar_h)

    # ---------- event handling ----------
    def on_resize(self, w, h):
        global WIDTH, HEIGHT, screen
        WIDTH, HEIGHT = w, h
        screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)

    def wheel(self, e):
        mods = pygame.key.get_mods()
        if mods & (pygame.KMOD_CTRL | pygame.KMOD_META):
            mx, my = pygame.mouse.get_pos()
            t_anchor = self.x_to_time(mx)
            if e.y > 0:
                self.set_zoom(self.px_per_sec * (1.15 ** e.y), anchor_time=t_anchor)
            else:
                self.set_zoom(self.px_per_sec / (1.15 ** (-e.y)), anchor_time=t_anchor)
        else:
            if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                self.h_off_sec = max(0.0, self.h_off_sec - e.y * (self.visible_secs() * 0.1))
            else:
                total_px = self.total_track_pixels()
                vis_px = HEIGHT - self.top_bar_h - self.ruler_h - self.bottom_bar_h
                max_off = max(0, total_px - vis_px)
                self.v_off_px = int(max(0, min(max_off, self.v_off_px - e.y * 60)))

    def key(self, e):
        if e.key == pygame.K_SPACE:
            if self.playing: self.stop()
            else: self.play()
        elif e.key in (pygame.K_DELETE, pygame.K_BACKSPACE):
            self.delete_selected()
        elif e.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.set_zoom(self.px_per_sec * 1.2)
        elif e.key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS):
            self.set_zoom(self.px_per_sec / 1.2)
        elif e.key == pygame.K_HOME:
            self.playhead = 0.0
        elif e.key == pygame.K_END:
            self.playhead = self.project_length()
        elif e.key == pygame.K_z:
            if self.selected_clip:
                l, r = self.selected_clip.bounds()
                mid = (l + r) / 2
                span = max(0.25, r - l)
                self.set_zoom(max(60, min(400, (WIDTH - self.timeline_origin_x) / span)), anchor_time=mid)
                self.h_off_sec = max(0.0, l - 0.25 * span)
        elif (e.key == pygame.K_s) and (pygame.key.get_mods() & (pygame.KMOD_CTRL | pygame.KMOD_META)):
            self.saveas()
        elif (e.key == pygame.K_e) and (pygame.key.get_mods() & (pygame.KMOD_CTRL | pygame.KMOD_META)):
            self.save_stems()

    def mouse_down(self, e):
        mx, my = e.pos
        self.drag_last_mouse = (mx, my)

        # Check if any button was clicked
        mouse_buttons = [self.btn_play, self.btn_stop, self.btn_tone, self.btn_save, self.btn_save_stems,
                         self.btn_add_track, self.btn_import, self.btn_delete,
                         self.btn_split, self.btn_zoom_in, self.btn_zoom_out, self.btn_snap]
        
        for button in mouse_buttons:
            if button.rect.collidepoint(mx, my):
                button.handle_click()
                return

        # click track header selects track
        top_tracks = self.top_bar_h + self.ruler_h
        if my >= top_tracks and mx < self.timeline_origin_x:
            ti = self._ensure_track_for_y(my)
            self.selected_track = ti
            # fader/mute/solo hit-testing (start fader drag)
            tr = self.tracks[ti]
            if tr._fader_rect and tr._fader_rect.collidepoint(mx, my):
                self.drag_mode = 'fader'; self.drag_fader_track = ti; return
            if tr._mute_rect and tr._mute_rect.collidepoint(mx, my):
                tr.mute = not tr.mute; self.invalidate_render(); return
            if tr._solo_rect and tr._solo_rect.collidepoint(mx, my):
                tr.solo = not tr.solo; self.invalidate_render(); return
            return

        # ruler scrub
        if self.top_bar_h <= my <= self.top_bar_h + self.ruler_h and mx >= self.timeline_origin_x:
            self.playhead = max(0.0, self.x_to_time(mx))
            self.drag_mode = 'scrub'
            return

        # scrollbars
        if hasattr(self, 'h_scroll_rect') and self.h_scroll_rect.collidepoint(mx, my):
            self._drag_scroll_h = True; return
        if hasattr(self, 'v_scroll_rect') and self.v_scroll_rect.collidepoint(mx, my):
            self._drag_scroll_v = True; return

        # right-button pan
        if e.button == 3:
            self.drag_mode = 'pan'; return

        # clips / lanes
        if mx >= self.timeline_origin_x and my >= self.top_bar_h + self.ruler_h:
            for ti in range(len(self.tracks)):
                tr = self.tracks[ti]
                for c in reversed(tr.clips):
                    if c.rect and c.rect.collidepoint(mx, my):
                        self.select_clip(c)
                        if mx - c.rect.x < 8: self.drag_mode = 'trim_l'
                        elif c.rect.right - mx < 8: self.drag_mode = 'trim_r'
                        elif my - c.rect.y < 14 and mx - c.rect.x < 14: self.drag_mode = 'fade_l'
                        elif my - c.rect.y < 14 and c.rect.right - mx < 14: self.drag_mode = 'fade_r'
                        else:
                            self.drag_mode = 'move'
                            self.drag_offset_time = self.x_to_time(mx) - c.start
                        self.drag_clip_ref = c
                        return
            # empty lane click → set playhead & select track
            self.playhead = self.maybe_snap(self.x_to_time(mx))
            self.selected_track = self._ensure_track_for_y(my)
            self.select_clip(None)

    def on_fader_drag(self, mx):
        tr = self.tracks[self.drag_fader_track]
        rel = (mx - tr._fader_rect.x) / max(1, tr._fader_rect.w - 1)
        tr.volume = max(0.0, min(1.5, rel))
        self.invalidate_render()

    def mouse_up(self, e):
        self.drag_mode = None
        self.drag_clip_ref = None
        self.drag_fader_track = None
        self._drag_scroll_h = False
        self._drag_scroll_v = False

    def mouse_motion(self, e):
        mx, my = e.pos
        dx = mx - self.drag_last_mouse[0]
        dy = my - self.drag_last_mouse[1]
        self.drag_last_mouse = (mx, my)

        if self.drag_mode == 'scrub':
            self.playhead = max(0.0, self.x_to_time(mx))
            return

        if self._drag_scroll_h:
            total = max(self.project_length(), self.visible_secs())
            vis = self.visible_secs()
            max_off = max(0.0, total - vis)
            track_w = WIDTH - self.timeline_origin_x - self.right_bar_w
            frac = vis / total if total else 1.0
            denom = max(1, int((1 - frac) * track_w))
            self.h_off_sec = max(0.0, min(max_off, self.h_off_sec + (dx / denom) * max_off))
            return

        if self._drag_scroll_v:
            total_px = self.total_track_pixels()
            vis_px = HEIGHT - self.top_bar_h - self.ruler_h - self.bottom_bar_h
            max_off = max(0, total_px - vis_px)
            track_h = HEIGHT - self.top_bar_h - self.ruler_h - self.bottom_bar_h
            frac = vis_px / total_px if total_px else 1.0
            denom = max(1, int((1 - frac) * track_h))
            self.v_off_px = int(max(0, min(max_off, self.v_off_px + (dy / denom) * max_off)))
            return

        if self.drag_mode == 'pan':
            self.h_off_sec = max(0.0, self.h_off_sec - dx / self.px_per_sec)
            total_px = self.total_track_pixels()
            vis_px = HEIGHT - self.top_bar_h - self.ruler_h - self.bottom_bar_h
            max_off = max(0, total_px - vis_px)
            self.v_off_px = int(max(0, min(max_off, self.v_off_px - dy)))
            return

        if self.drag_mode in ('move', 'trim_l', 'trim_r', 'fade_l', 'fade_r'):
            c = self.drag_clip_ref
            if not c:
                return
            if self.drag_mode == 'move':
                t_raw = self.x_to_time(mx) - self.drag_offset_time
                c.start = self.maybe_snap(t_raw)
                # move between tracks (auto-create if needed)
                cand = self._ensure_track_for_y(my)
                if cand != c.track_index:
                    if c in self.tracks[c.track_index].clips:
                        self.tracks[c.track_index].clips.remove(c)
                    c.track_index = cand
                    self.tracks[cand].clips.append(c)
            elif self.drag_mode == 'trim_l':
                t = self.x_to_time(mx)
                left, right = c.bounds()
                t = max(min(t, right - 0.05), 0.0)
                src_new = c.in_pos + (t - left) * c.speed
                src_new = max(0.0, min(src_new, c.out_pos - 0.01))
                c.in_pos = src_new
                c.start = self.maybe_snap(t)
            elif self.drag_mode == 'trim_r':
                t = self.x_to_time(mx)
                left, _ = c.bounds()
                t = max(t, left + 0.05)
                src_new = c.in_pos + (t - c.start) * c.speed
                src_new = max(c.in_pos + 0.01, min(src_new, len(c.audio)/SR))
                c.out_pos = src_new
            elif self.drag_mode == 'fade_l':
                c.fade_in = max(0.0, min(3.0, c.fade_in + dx / 200.0))
            elif self.drag_mode == 'fade_r':
                c.fade_out = max(0.0, min(3.0, c.fade_out - dx / 200.0))
            if c is self.selected_clip:
                self.sld_gain.val = c.gain
                self.sld_speed.val = c.speed
            self.invalidate_render()

        if self.drag_mode == 'fader' and self.drag_fader_track is not None:
            self.on_fader_drag(mx)

    # ---------- update loop ----------
    def update(self, dt_ms):
        if self.selected_clip and self.drag_mode is None:
            g = float(self.sld_gain.val)
            s = float(self.sld_speed.val)
            if abs(self.selected_clip.gain - g) > 1e-6 or abs(self.selected_clip.speed - s) > 1e-6:
                self.selected_clip.gain = g
                self.selected_clip.speed = s
                self.invalidate_render()
        self.bpm = float(self.sld_bpm.val)
        if self.playing and self.play_start_clock is not None:
            elapsed = time.perf_counter() - self.play_start_clock
            self.playhead = self.play_start_pos + elapsed
            if self.play_end_sec is not None and self.playhead >= self.play_end_sec:
                self.stop()
            if self.playhead > self.h_off_sec + 0.85 * self.visible_secs():
                self.h_off_sec = max(0.0, self.playhead - 0.85 * self.visible_secs())
        self.update_meters()

    # ---------- main event hook ----------
    def handle_event(self, e):
        if e.type == pygame.MOUSEBUTTONDOWN:
            self.mouse_down(e)
        elif e.type == pygame.MOUSEBUTTONUP:
            self.mouse_up(e)
        elif e.type == pygame.MOUSEMOTION and e.buttons != (0, 0, 0):
            self.mouse_motion(e)
        elif e.type == pygame.MOUSEWHEEL:
            self.wheel(e)
        elif e.type == pygame.KEYDOWN:
            self.key(e)
        elif e.type == pygame.VIDEORESIZE:
            self.on_resize(e.w, e.h)


def main():
    app = SimpleEditor()

    # Allow drop to import
    try:
        pygame.event.set_allowed([pygame.QUIT, pygame.DROPFILE, pygame.MOUSEBUTTONDOWN,
                                  pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION, pygame.MOUSEWHEEL,
                                  pygame.KEYDOWN, pygame.VIDEORESIZE])
    except Exception:
        pass

    # Preload from CLI
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        y, _ = load_audio(sys.argv[1])
        if y.size > 0:
            c = Clip(y, os.path.basename(sys.argv[1]))
            c.track_index = 0
            app.tracks[0].clips.append(c)
            app.select_clip(c)
        else:
            app.set_status(f"Failed to load audio from {sys.argv[1]}", ok=False)

    clock = pygame.time.Clock()
    while True:
        dt = clock.tick(60)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if e.type == pygame.DROPFILE:
                mx, my = pygame.mouse.get_pos()
                app._import_at_screen_pos(e.file, mx, my)
                continue
            app.handle_event(e)

        app.update(dt)
        app.draw(screen)
        pygame.display.flip()


if __name__ == "__main__":
    main()
