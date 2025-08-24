import os
import sys
import wave
import warnings
import subprocess
from math import log10
from typing import List, Tuple, Optional

import numpy as np
import pygame

# -------- Optional loader / stretch (no DawDreamer) --------
try:
    import librosa
    import soundfile as sf  # noqa: F401
    LIBROSA_AVAILABLE = True
except Exception:
    LIBROSA_AVAILABLE = False

SR = 44100
warnings.filterwarnings("ignore", message=r".*__audioread_load.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=r"PySoundFile failed.*")

# -------- macOS-safe dialogs (avoid Tk/SDL clash) --------
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


# -------- Audio helpers --------
def load_audio(path: str, target_sr: int = SR) -> Tuple[np.ndarray, int]:
    """Return mono float32 [-1,1], sr."""
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


def time_stretch(y: np.ndarray, speed: float) -> np.ndarray:
    """speed>1 → faster (shorter). If no librosa, naive resample (pitch changes)."""
    speed = max(0.25, min(2.0, float(speed)))
    if not y.size:
        return y.astype(np.float32)
    if LIBROSA_AVAILABLE:
        return librosa.effects.time_stretch(y, rate=speed).astype(np.float32)
    n_new = max(1, int(len(y) / speed))
    t_old = np.linspace(0, 1, len(y), endpoint=False)
    t_new = np.linspace(0, 1, n_new, endpoint=False)
    return np.interp(t_new, t_old, y).astype(np.float32)


def to_stereo(mono: np.ndarray) -> np.ndarray:
    return np.vstack([mono, mono])


def to_int16_stereo(st: np.ndarray) -> np.ndarray:
    st = np.clip(st, -1.0, 1.0)
    return (st * 32767).astype(np.int16)


# -------- Data models --------
class Clip:
    def __init__(self, audio: np.ndarray, name: str, start=0.0, speed=1.0, gain=1.0, in_pos=0.0, out_pos=None):
        self.audio = audio.astype(np.float32)
        self.name = name
        self.start = float(start)    # timeline start (sec)
        self.speed = float(speed)    # 1.0 normal; 0.5 slower; 1.5 faster
        self.gain = float(gain)
        self.in_pos = float(in_pos)  # seconds into source
        self.out_pos = float(out_pos) if out_pos is not None else (len(audio)/SR)
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
                          speed=self.speed, gain=self.gain, in_pos=split_src, out_pos=self.out_pos)
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


# -------- UI setup --------
pygame.init()
pygame.mixer.init(frequency=SR, size=-16, channels=2, buffer=1024)

WIDTH, HEIGHT = 1220, 860
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Simple Audio Editor — Multi-Track + Meters (v4)")

COL_BG = (20, 20, 26)
COL_PANEL = (36, 36, 48)
COL_ACC = (95, 175, 255)
COL_ACC2 = (120, 220, 160)
COL_TXT = (235, 235, 235)
COL_MUTED = (150, 150, 160)
COL_ERR = (240, 80, 80)
COL_OK = (120, 220, 120)
COL_GRID = (70, 70, 85)
COL_CLIP = (85, 120, 200)
COL_CLIP_SEL = (180, 140, 80)
COL_METER_BG = (40, 40, 55)
COL_METER = (90, 220, 120)
COL_MUTE = (210, 80, 80)
COL_SOLO = (240, 200, 90)

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

    def handle(self, e):
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and self.hover:
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
        rel = (self.val - self.min) / (self.max - self.min)
        x = self.rect.x + int(rel * self.rect.w)
        pygame.draw.circle(surf, COL_ACC, (x, self.rect.y + 10), 8)

    def update(self, mouse, pressed):
        if pressed[0]:
            if self.drag or self.rect.collidepoint(mouse):
                self.drag = True
                rel = (mouse[0] - self.rect.x) / self.rect.w
                rel = max(0, min(1, rel))
                self.val = self.min + rel * (self.max - self.min)
        else:
            self.drag = False


# -------- Editor --------
class SimpleEditor:
    def __init__(self):
        # layout
        self.timeline_origin_x = 210
        self.track_h = 92
        self.ruler_h = 34
        self.top_bar_h = 60
        self.px_per_sec = 90.0
        self.snap = True         # snap to 0.1 s grid when dragging horizontally
        self.snap_sec = 0.1

        # model
        self.tracks: List[Track] = []
        self._create_track()                  # Track 1
        self.selected_track = 0
        self.selected_clip: Optional[Clip] = None

        # playback
        self.playhead = 0.0
        self.playing = False
        self.play_started_ms: Optional[int] = None
        self.render_start_sec = 0.0
        self.play_end_sec: Optional[float] = None
        self._stems: Optional[List[np.ndarray]] = None
        self.mix_sound: Optional[pygame.mixer.Sound] = None

        # drag state
        self.drag_mode: Optional[str] = None  # 'move','trim_l','trim_r','scrub','fader'
        self.drag_offset_time = 0.0
        self.drag_clip_ref: Optional[Clip] = None
        self.drag_fader_track: Optional[int] = None
        self.drag_from_track: Optional[int] = None

        # UI
        self.btn_play = Button(10, 10, 60, 30, "Play", self.play)
        self.btn_stop = Button(80, 10, 60, 30, "Stop", self.stop)
        self.btn_save = Button(150, 10, 110, 30, "Save WAV…", self.saveas)
        self.btn_add_track = Button(270, 10, 90, 30, "+ Track", self.add_track)
        self.btn_import = Button(365, 10, 120, 30, "Import Clip", self.import_clip)
        self.btn_delete = Button(490, 10, 90, 30, "Delete", self.delete_selected)
        self.btn_split = Button(585, 10, 120, 30, "Split @ Play", self.split_at_playhead)
        self.btn_zoom_in = Button(710, 10, 60, 30, "+", lambda: self.set_zoom(self.px_per_sec * 1.25))
        self.btn_zoom_out = Button(775, 10, 60, 30, "-", lambda: self.set_zoom(self.px_per_sec / 1.25))
        self.btn_snap = Button(840, 10, 80, 30, "Snap", self.toggle_snap, toggle=True, state=self.snap)

        self.sld_gain = Slider(10, self.top_bar_h + 6, 170, "Clip Gain", 0.0, 2.0, 1.0)
        self.sld_speed = Slider(10, self.top_bar_h + 36, 170, "Clip Speed", 0.5, 1.5, 1.0)

        self.status = "Ready."
        self.status_col = COL_TXT

    # ----- track creation / naming -----
    def _create_track(self):
        idx = len(self.tracks) + 1             # 1-based numbering
        self.tracks.append(Track(f"Track {idx}"))

    def add_track(self):
        self._create_track()
        self.selected_track = len(self.tracks) - 1
        self.set_status(f"Added Track {self.selected_track + 1}")

    # ----- helpers -----
    def set_status(self, msg, ok=True):
        self.status = msg
        self.status_col = COL_OK if ok else COL_ERR

    def set_zoom(self, new_pps: float):
        self.px_per_sec = max(30.0, min(480.0, float(new_pps)))

    def maybe_snap(self, t: float) -> float:
        if not self.snap:
            return max(0.0, t)
        grid = self.snap_sec
        return max(0.0, round(t / grid) * grid)

    def time_to_x(self, t_sec: float) -> int:
        return int(self.timeline_origin_x + t_sec * self.px_per_sec)

    def x_to_time(self, x: int) -> float:
        return max(0.0, (x - self.timeline_origin_x) / self.px_per_sec)

    def track_y(self, idx: int) -> int:
        return self.top_bar_h + self.ruler_h + idx * self.track_h

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

    # ----- selection -----
    def select_clip(self, clip: Optional[Clip]):
        if self.selected_clip is not None:
            self.selected_clip.selected = False
        self.selected_clip = clip
        if clip is not None:
            clip.selected = True
            if self.drag_mode is None:
                self.sld_gain.val = clip.gain
                self.sld_speed.val = clip.speed

    # ----- clip ops -----
    def import_clip(self):
        path = choose_open_file("Choose audio")
        if not path:
            self.set_status("Import cancelled.", ok=True)
            return
        try:
            y, _ = load_audio(path)
            name = os.path.basename(path)
            c = Clip(y, name=name, start=self.playhead, speed=1.0, gain=1.0)
            c.track_index = self.selected_track
            self.tracks[self.selected_track].clips.append(c)
            self.select_clip(c)
            self.set_status(f"Imported {name} → Track {self.selected_track+1}")
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
        self.set_status("Clip deleted.")

    def split_at_playhead(self):
        c = self.selected_clip
        if not c:
            return
        rc = c.cut_at(self.playhead)
        if rc is not None:
            rc.track_index = c.track_index
            self.tracks[c.track_index].clips.append(rc)
            self.set_status("Clip split.")
        else:
            self.set_status("Playhead not inside clip.", ok=False)

    # ----- rendering / stems -----
    def render_mix_with_stems(self, start_sec: float = 0.0) -> Tuple[np.ndarray, float, List[np.ndarray]]:
        end_sec = self.project_length()
        if end_sec <= start_sec + 1e-6:
            return np.zeros(1, dtype=np.float32), start_sec, [np.zeros(1, dtype=np.float32) for _ in self.tracks]
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
                seg = c.audio[int(c.in_pos * SR): int(c.out_pos * SR)]
                if seg.size == 0:
                    continue
                stretched = time_stretch(seg, c.speed)
                clip_start = c.start
                place_start = max(0.0, clip_start - start_sec)
                i0 = int(place_start * SR)
                if clip_start < start_sec:
                    cut = int((start_sec - clip_start) * SR)
                    stretched = stretched[cut:] if cut < len(stretched) else np.zeros(0, dtype=np.float32)
                if stretched.size == 0 or i0 >= n_total:
                    continue
                take = min(n_total - i0, len(stretched))
                if take > 0:
                    stem[i0:i0+take] += stretched[:take] * c.gain
            stems[ti] = stems[ti] * tr.volume

        mix = np.sum(stems, axis=0)
        mix = np.clip(mix, -1.0, 1.0)
        return mix, start_sec, stems

    def play(self):
        audio, start, stems = self.render_mix_with_stems(self.playhead)
        if audio.size <= 1:
            self.set_status("Nothing to play.", ok=False)
            return
        st = to_int16_stereo(to_stereo(audio))
        samples = np.transpose(st)
        self.mix_sound = pygame.sndarray.make_sound(samples.copy())
        self.mix_sound.play()
        self.playing = True
        self.play_started_ms = pygame.time.get_ticks()
        self.render_start_sec = start
        self.play_end_sec = self.project_length()
        self._stems = stems
        self.set_status("Playing…")

    def stop(self):
        pygame.mixer.stop()
        self.playing = False
        self.play_started_ms = None
        self._stems = None
        self.set_status("Stopped.")

    def saveas(self):
        audio, _, _ = self.render_mix_with_stems(0.0)
        if audio.size <= 1:
            self.set_status("Nothing to save.", ok=False)
            return
        path = choose_save_file("Save Mix As", default_name="mix.wav")
        if not path:
            self.set_status("Save cancelled.", ok=True)
            return
        try:
            st = to_int16_stereo(to_stereo(audio))
            with wave.open(path, 'wb') as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(SR)
                wf.writeframes(st.T.astype('<i2').tobytes())
            self.set_status(f"Saved: {path}")
        except Exception as e:
            self.set_status(f"Save failed: {e}", ok=False)

    # ----- hit-testing -----
    def pos_to_clip(self, x: int, y: int) -> Tuple[Optional[Clip], Optional[pygame.Rect]]:
        if y < self.top_bar_h + self.ruler_h:
            return None, None
        row = int((y - (self.top_bar_h + self.ruler_h)) // self.track_h)
        if row < 0 or row >= len(self.tracks):
            return None, None
        t = self.tracks[row]
        for c in sorted(t.clips, key=lambda k: k.start, reverse=True):
            x1 = self.time_to_x(c.start)
            x2 = self.time_to_x(c.start + max(0.05, c.eff_len_sec))
            r = pygame.Rect(x1, self.track_y(row) + 6, x2 - x1, self.track_h - 12)
            if r.collidepoint(x, y):
                return c, r
        return None, None

    # ----- mouse -----
    def on_mousedown(self, pos, button):
        x, y = pos

        # Track header (mute/solo/fader)
        if y >= self.top_bar_h + self.ruler_h:
            row = int((y - (self.top_bar_h + self.ruler_h)) // self.track_h)
            if 0 <= row < len(self.tracks) and x < self.timeline_origin_x:
                self.selected_track = row
                tr = self.tracks[row]
                if tr._mute_rect and tr._mute_rect.collidepoint(x, y):
                    tr.mute = not tr.mute; self.set_status(f"{tr.name} mute = {tr.mute}"); return
                if tr._solo_rect and tr._solo_rect.collidepoint(x, y):
                    tr.solo = not tr.solo; self.set_status(f"{tr.name} solo = {tr.solo}"); return
                if tr._fader_rect and tr._fader_rect.collidepoint(x, y):
                    self.drag_mode = 'fader'; self.drag_fader_track = row; self._apply_fader_from_x(x); return

        # Ruler scrubbing
        if self.top_bar_h <= y <= self.top_bar_h + self.ruler_h:
            self.playhead = self.x_to_time(x)
            self.drag_mode = 'scrub'
            return

        # Clips
        c, rect = self.pos_to_clip(x, y)
        if c is not None:
            self.selected_track = c.track_index
            self.select_clip(c)
            handle = 8
            if abs(x - rect.left) <= handle:
                self.drag_mode = 'trim_l'
            elif abs(x - rect.right) <= handle:
                self.drag_mode = 'trim_r'
            else:
                self.drag_mode = 'move'
                self.drag_offset_time = self.x_to_time(x) - c.start
            self.drag_clip_ref = c
            self.drag_from_track = c.track_index
            return

        # Empty track area selects track
        row = int((y - (self.top_bar_h + self.ruler_h)) // self.track_h)
        if 0 <= row < len(self.tracks):
            self.selected_track = row
            self.select_clip(None)

    def _apply_fader_from_x(self, x: int):
        ti = self.drag_fader_track
        if ti is None:
            return
        tr = self.tracks[ti]
        fr = tr._fader_rect
        if not fr:
            return
        rel = (x - fr.x) / max(1, fr.w)
        rel = max(0.0, min(1.0, rel))
        tr.volume = 2.0 * rel  # 0..2x

    def on_mouseup(self, pos, button):
        self.drag_mode = None
        self.drag_clip_ref = None
        self.drag_fader_track = None
        self.drag_from_track = None

    def on_mousemove(self, pos, buttons):
        x, y = pos
        if self.drag_mode == 'scrub' and buttons[0]:
            self.playhead = self.x_to_time(x)
            return
        if self.drag_mode == 'fader' and buttons[0]:
            self._apply_fader_from_x(x)
            return
        c = self.drag_clip_ref
        if c is None:
            return

        # Determine target track under pointer
        area_top = self.top_bar_h + self.ruler_h
        row = int((y - area_top) // self.track_h)

        if self.drag_mode == 'move' and buttons[0]:
            # Horizontal
            new_start = self.maybe_snap(self.x_to_time(x) - self.drag_offset_time)
            c.start = max(0.0, new_start)

            # Vertical: move clip between track lists (FIX)
            if 0 <= row < len(self.tracks) and row != c.track_index:
                # remove from old track list
                old_idx = c.track_index
                if c in self.tracks[old_idx].clips:
                    self.tracks[old_idx].clips.remove(c)
                # add to new track
                c.track_index = row
                self.tracks[row].clips.append(c)
                self.selected_track = row

        elif self.drag_mode == 'trim_l' and buttons[0]:
            new_left_t = max(0.0, self.x_to_time(x))
            if self.snap:
                new_left_t = self.maybe_snap(new_left_t)
            dt = new_left_t - c.start
            if dt != 0.0:
                src_advance = dt * c.speed
                new_in = min(c.out_pos - 0.01, max(0.0, c.in_pos + src_advance))
                applied = new_in - c.in_pos
                c.in_pos = new_in
                c.start = c.start + (applied / max(c.speed, 1e-6))

        elif self.drag_mode == 'trim_r' and buttons[0]:
            new_right_t = max(c.start + 0.01, self.x_to_time(x))
            if self.snap:
                new_right_t = max(c.start + 0.01, self.maybe_snap(new_right_t))
            eff_len = new_right_t - c.start
            c.out_pos = c.in_pos + eff_len * c.speed

    # ----- keyboard -----
    def handle_keys(self, e):
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_SPACE:
                if self.playing: self.stop()
                else: self.play()
            elif e.key == pygame.K_s and (e.mod & pygame.KMOD_META or e.mod & pygame.KMOD_CTRL):
                self.saveas()
            elif e.key in (pygame.K_DELETE, pygame.K_BACKSPACE):
                self.delete_selected()

    # ----- meters -----
    def _db_to_level(self, db: float) -> float:
        return max(0.0, min(1.0, (db + 60.0) / 60.0))  # map −60..0 dB → 0..1

    def _update_track_meters(self, now_sec: float):
        if not self.playing or self._stems is None:
            for tr in self.tracks:
                tr.meter_level *= 0.85
            return
        idx = int((now_sec - self.render_start_sec) * SR)
        win = int(0.05 * SR)  # 50 ms window
        for ti, tr in enumerate(self.tracks):
            stem = self._stems[ti] if 0 <= ti < len(self._stems) else None
            if stem is None or idx >= len(stem):
                tr.meter_level *= 0.85
                continue
            i0 = max(0, idx - win)
            i1 = min(len(stem), idx + win)
            sl = stem[i0:i1]
            if sl.size <= 4:
                tr.meter_level *= 0.85
                continue
            rms = float(np.sqrt(np.mean(sl * sl)) + 1e-8)
            db = 20.0 * log10(rms)
            level = self._db_to_level(db)
            tr.meter_level = max(level, tr.meter_level * 0.85)

    # ----- drawing -----
    def draw_ruler(self, surf, width):
        y0 = self.top_bar_h
        pygame.draw.rect(surf, COL_PANEL, pygame.Rect(0, y0, width, self.ruler_h))
        max_sec = max(10, int((width - self.timeline_origin_x) / self.px_per_sec) + 5)
        for s in range(max_sec):
            x = self.time_to_x(s)
            pygame.draw.line(surf, COL_GRID, (x, y0), (x, y0 + self.ruler_h))
            t = FONT_SM.render(str(s), True, COL_MUTED)
            surf.blit(t, (x + 2, y0 + 8))
        px = self.time_to_x(self.playhead)
        pygame.draw.line(surf, (250, 90, 90), (px, y0), (px, HEIGHT), 2)

    def draw_tracks(self, surf, width, height):
        # headers + meters/faders
        for i, tr in enumerate(self.tracks):
            y = self.track_y(i)
            pygame.draw.rect(surf, COL_PANEL, pygame.Rect(0, y, self.timeline_origin_x - 2, self.track_h))
            pygame.draw.rect(surf, (60, 60, 80), pygame.Rect(0, y, self.timeline_origin_x - 2, self.track_h), 2)
            surf.blit(FONT_MD.render(tr.name, True, COL_TXT), (10, y + 6))

            # mute / solo
            tr._mute_rect = pygame.Rect(10, y + 30, 44, 22)
            tr._solo_rect = pygame.Rect(60, y + 30, 44, 22)
            pygame.draw.rect(surf, COL_MUTE if tr.mute else (90, 90, 90), tr._mute_rect, border_radius=5)
            pygame.draw.rect(surf, (50, 50, 50), tr._mute_rect, 2, border_radius=5)
            pygame.draw.rect(surf, COL_SOLO if tr.solo else (90, 90, 90), tr._solo_rect, border_radius=5)
            pygame.draw.rect(surf, (50, 50, 50), tr._solo_rect, 2, border_radius=5)
            surf.blit(FONT_SM.render("M", True, (15, 15, 18)), tr._mute_rect.move(15, 3))
            surf.blit(FONT_SM.render("S", True, (15, 15, 18)), tr._solo_rect.move(15, 3))

            # fader
            f_x = 112
            tr._fader_rect = pygame.Rect(f_x, y + 32, 80, 8)
            pygame.draw.rect(surf, (70, 70, 90), tr._fader_rect, border_radius=4)
            rel = tr.volume / 2.0
            cx = int(tr._fader_rect.x + rel * tr._fader_rect.w)
            pygame.draw.circle(surf, COL_ACC, (cx, tr._fader_rect.y + tr._fader_rect.h // 2), 7)
            surf.blit(FONT_SM.render(f"{tr.volume:.2f}x", True, COL_MUTED), (f_x + 86, y + 23))

            # meter
            m_w = 10; m_h = self.track_h - 12
            mx = self.timeline_origin_x - 16; my = y + 6
            pygame.draw.rect(surf, COL_METER_BG, pygame.Rect(mx, my, m_w, m_h), border_radius=3)
            lvl = int(m_h * max(0.0, min(1.0, tr.meter_level)))
            if lvl > 0:
                pygame.draw.rect(surf, COL_METER, pygame.Rect(mx, my + (m_h - lvl), m_w, lvl), border_radius=3)

        # timeline rows + clips
        area = pygame.Rect(self.timeline_origin_x, self.top_bar_h + self.ruler_h,
                           width - self.timeline_origin_x, height - (self.top_bar_h + self.ruler_h))
        pygame.draw.rect(surf, (28, 28, 38), area)
        for i, tr in enumerate(self.tracks):
            y = self.track_y(i)
            pygame.draw.line(surf, (50, 50, 64), (self.timeline_origin_x, y), (width, y))
            for c in tr.clips:
                x1 = self.time_to_x(c.start)
                w = max(10, int(max(0.02, c.eff_len_sec) * self.px_per_sec))
                rect = pygame.Rect(x1, y + 6, w, self.track_h - 12)
                c.rect = rect
                col = COL_CLIP_SEL if c.selected else COL_CLIP
                pygame.draw.rect(surf, col, rect, border_radius=6)
                pygame.draw.rect(surf, (col[0]//2, col[1]//2, col[2]//2), rect, 2, border_radius=6)
                pygame.draw.rect(surf, (240, 240, 240), pygame.Rect(rect.left - 2, rect.top, 4, rect.h))
                pygame.draw.rect(surf, (240, 240, 240), pygame.Rect(rect.right - 2, rect.top, 4, rect.h))
                label = f"{c.name}  x{c.speed:.2f}  g{c.gain:.2f}"
                surf.blit(FONT_SM.render(label, True, (15, 15, 18)), (rect.x + 6, rect.y + 4))

    def draw_topbar(self, surf, width):
        pygame.draw.rect(surf, COL_PANEL, pygame.Rect(0, 0, width, self.top_bar_h))
        for b in [self.btn_play, self.btn_stop, self.btn_save, self.btn_add_track,
                  self.btn_import, self.btn_delete, self.btn_split, self.btn_zoom_in, self.btn_zoom_out, self.btn_snap]:
            b.draw(surf)
        surf.blit(FONT_SM.render(self.status, True, self.status_col), (width - 360, 18))

    # ----- main loop -----
    def run(self):
        clock = pygame.time.Clock()
        global WIDTH, HEIGHT, screen
        try:
            pygame.event.set_allowed([
                pygame.QUIT, pygame.DROPFILE, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP,
                pygame.MOUSEMOTION, pygame.KEYDOWN, pygame.VIDEORESIZE
            ])
        except Exception:
            pass

        running = True
        last_tick_ms = pygame.time.get_ticks()

        while running:
            now = pygame.time.get_ticks()
            dt_sec = (now - last_tick_ms) / 1000.0
            last_tick_ms = now

            mouse = pygame.mouse.get_pos()
            buttons = pygame.mouse.get_pressed()

            # Button hovers
            for b in [self.btn_play, self.btn_stop, self.btn_save, self.btn_add_track,
                      self.btn_import, self.btn_delete, self.btn_split, self.btn_zoom_in, self.btn_zoom_out, self.btn_snap]:
                b.update(mouse)

            # Sliders only when NOT dragging clips/faders
            if self.drag_mode is None and self.selected_clip is not None:
                self.sld_gain.update(mouse, buttons)
                self.sld_speed.update(mouse, buttons)
                self.selected_clip.gain = self.sld_gain.val
                self.selected_clip.speed = self.sld_speed.val

            # Events
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.VIDEORESIZE:
                    WIDTH, HEIGHT = e.w, e.h
                    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
                elif e.type in (pygame.KEYDOWN, pygame.KEYUP):
                    self.handle_keys(e)
                elif e.type == pygame.DROPFILE:
                    path = e.file
                    try:
                        y, _ = load_audio(path)
                        name = os.path.basename(path)
                        c = Clip(y, name=name, start=self.playhead, speed=1.0, gain=1.0)
                        c.track_index = self.selected_track
                        self.tracks[self.selected_track].clips.append(c)
                        self.select_clip(c)
                        self.set_status(f"Imported (drop) {name} → Track {self.selected_track+1}")
                    except Exception as ex:
                        self.set_status(f"Drop import failed: {ex}", ok=False)
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    self.on_mousedown(e.pos, e.button)
                    for b in [self.btn_play, self.btn_stop, self.btn_save, self.btn_add_track,
                              self.btn_import, self.btn_delete, self.btn_split, self.btn_zoom_in, self.btn_zoom_out, self.btn_snap]:
                        b.handle(e)
                elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                    self.on_mouseup(e.pos, e.button)
                elif e.type == pygame.MOUSEMOTION:
                    self.on_mousemove(e.pos, buttons)

            # Playback timeline + meters (playhead always advances while playing)
            if self.playing and self.play_started_ms is not None:
                self.playhead += dt_sec
                # update meters at current time
                self._update_track_meters(self.render_start_sec + self.playhead - self.render_start_sec)
                if self.play_end_sec is not None and self.playhead >= self.play_end_sec - 1e-3:
                    self.stop()
            else:
                self._update_track_meters(self.playhead)

            # Draw
            screen.fill(COL_BG)
            self.draw_topbar(screen, WIDTH)
            pygame.draw.rect(screen, COL_PANEL,
                             pygame.Rect(0, self.top_bar_h, 190, self.ruler_h + len(self.tracks)*self.track_h))
            self.sld_gain.draw(screen); self.sld_speed.draw(screen)
            self.draw_ruler(screen, WIDTH)
            self.draw_tracks(screen, WIDTH, HEIGHT)
            pygame.display.flip()

        pygame.quit()


def main():
    app = SimpleEditor()
    # Optional CLI import
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        try:
            y, _ = load_audio(sys.argv[1])
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
