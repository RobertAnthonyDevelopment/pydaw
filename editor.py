import os
import sys
import time
import math
import pickle
import pygame
import numpy as np
from typing import List, Optional, Dict, Tuple

from audio_utils import (
    load_audio,
    time_stretch,
    to_stereo,
    to_int16_stereo,
    apply_fades,
    SR,
)
from models import Clip, Track
from ui_components import (
    Button,
    Slider,
    TextInput,
    COL_TXT,
    COL_ERR,
    COL_OK,
    COL_PANEL,
    COL_GRID,
    COL_GRID_BAR,
    COL_CLIP,
    COL_CLIP_SEL,
    COL_FADE,
    COL_METER_BG,
    COL_METER,
    COL_SCROLL,
    COL_SCROLL_KNOB,
    COL_ACC,
    COL_ACC2,
    COL_BPM_BG,
    FONT_SM,
    FONT_MD,
)


# ---------- Simple pickers for .pdaw ----------

def _choose_open_project() -> Optional[str]:
    if sys.platform == "darwin":
        try:
            import subprocess
            osa = 'POSIX path of (choose file with prompt "Open Project" of type {"public.data"})'
            res = subprocess.run(["osascript", "-e", osa], capture_output=True, text=True)
            out = res.stdout.strip()
            return out or None
        except Exception:
            pass
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.update()
        fp = filedialog.askopenfilename(
            title="Open Project",
            filetypes=[("PyDAW Project", "*.pdaw"), ("All files", "*.*")]
        )
        root.destroy()
        return fp or None
    except Exception:
        return None


def _choose_save_project(default_name: str = "project.pdaw") -> Optional[str]:
    if sys.platform == "darwin":
        try:
            import subprocess
            osa = f'POSIX path of (choose file name with prompt "Save Project As" default name "{default_name}")'
            res = subprocess.run(["osascript", "-e", osa], capture_output=True, text=True)
            path = res.stdout.strip() if res.returncode == 0 else None
            if path and not path.lower().endswith(".pdaw"):
                path += ".pdaw"
            return path
        except Exception:
            pass
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.update()
        fp = filedialog.asksaveasfilename(
            title="Save Project As",
            defaultextension=".pdaw",
            filetypes=[("PyDAW Project", "*.pdaw"), ("All files", "*.*")],
            initialfile=default_name
        )
        root.destroy()
        return fp or None
    except Exception:
        return None


# =====================
#       Editor
# =====================

class SimpleEditor:
    def __init__(self):
        pygame.init()
        pygame.sndarray.use_arraytype("numpy")

        self.WIDTH, self.HEIGHT = 1280, 860
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("PyDAW - Python Digital Audio Workstation")

        self._init_mixer()

        # Layout metrics
        self.BUTTON_W = 90
        self.BUTTON_H = 32
        self.BUTTON_GAP_X = 10
        self.ROW_GAP_Y = 12
        self.TOP_PAD = 10
        self.BOTTOM_PAD = 10
        self.TOPBAR_ROWS = 4
        self.top_bar_h = (
            self.TOP_PAD
            + self.TOPBAR_ROWS * self.BUTTON_H
            + (self.TOPBAR_ROWS - 1) * self.ROW_GAP_Y
            + self.BOTTOM_PAD
        )

        self.ruler_h = 36
        self.bottom_bar_h = 24

        # Scrollbars
        self.scrollbar_h = 16
        self.scrollbar_w = 16
        self.right_bar_w = self.scrollbar_w

        # Panels
        self.MIN_LEFT_PANEL = 220
        self.MAX_LEFT_PANEL_RATIO = 0.45  # of window width
        self.track_header_w = 260

        # Left panel (user-resizable via splitter)
        self.left_panel_w = 300
        self.user_left_panel_w: Optional[int] = None  # remember user width across window resizes
        self.SPLIT_HIT_PAD = 6
        self.splitter_drag = False
        self._splitter_grab_dx = 0

        # Derived
        self.timeline_origin_x = self.left_panel_w + self.track_header_w
        self.track_h = 100

        # View / scroll
        self.px_per_sec = 100.0
        self.h_off_sec = 0.0
        self.v_off_px = 0
        self._drag_scroll_h = False
        self._drag_scroll_v = False
        self.h_bar_rect = pygame.Rect(0, 0, 0, 0)
        self.v_bar_rect = pygame.Rect(0, 0, 0, 0)
        self.h_scroll_rect = pygame.Rect(0, 0, 0, 0)
        self.v_scroll_rect = pygame.Rect(0, 0, 0, 0)

        # Grid / snapping
        self.bpm = 120.0
        self.subdiv = 4
        self.snap = True

        # Model
        self.tracks: List[Track] = []
        self._create_track()
        self._create_track()
        self.selected_track = 0
        self.selected_clip: Optional[Clip] = None

        # Playback state
        self.playhead = 0.0
        self.playing = False
        self.paused = False
        self.looping = False
        self.play_start_clock: Optional[float] = None
        self.play_start_pos: float = 0.0
        self.play_end_sec: Optional[float] = None
        self._stems: Optional[List[np.ndarray]] = None
        self.mix_sound: Optional[pygame.mixer.Sound] = None
        self.play_channel: Optional[pygame.mixer.Channel] = None

        # Dragging
        self.drag_mode: Optional[str] = None
        self.drag_offset_time = 0.0
        self.drag_clip_ref: Optional[Clip] = None
        self.drag_fader_track: Optional[int] = None
        self.drag_last_mouse = (0, 0)

        # Project / cache
        self.project_path: Optional[str] = None
        self.project_modified = False
        self.audio_files: Dict[str, np.ndarray] = {}

        # UI
        self._create_ui_elements()

        self.status = "Ready."
        self.status_col = COL_TXT

    # ---------- Init ----------

    def _init_mixer(self):
        ok = False
        for buffer_size in [1024, 2048, 4096, 8192]:
            try:
                pygame.mixer.init(frequency=SR, size=-16, channels=2, buffer=buffer_size)
                pygame.mixer.set_num_channels(64)
                ok = True
                break
            except pygame.error:
                pass
        if not ok:
            try:
                pygame.mixer.init()
            except pygame.error as e:
                print(f"Failed to initialize mixer: {e}")
                sys.exit(1)

    # ---------- UI creation & reflow ----------

    def _create_ui_elements(self):
        self.top_bar_h = (
            self.TOP_PAD
            + self.TOPBAR_ROWS * self.BUTTON_H
            + (self.TOPBAR_ROWS - 1) * self.ROW_GAP_Y
            + self.BOTTOM_PAD
        )

        def add_row(button_defs, row_index: int, start_x=10):
            y = self.TOP_PAD + row_index * (self.BUTTON_H + self.ROW_GAP_Y)
            x = start_x
            made = []
            for label, fn, opts in button_defs:
                w = opts.get("w", self.BUTTON_W)
                toggle = opts.get("toggle", False)
                state = opts.get("state", False)
                made.append(Button(x, y, w, self.BUTTON_H, label, fn, toggle=toggle, state=state))
                x += w + self.BUTTON_GAP_X
            return made

        # Rows
        row1 = [
            ("Play", self.play, {}),
            ("Pause", self.pause, {}),
            ("Stop", self.stop, {}),
            ("Loop", self.toggle_loop, {"toggle": True, "state": self.looping}),
            ("Rewind", self.rewind, {}),
            ("Fast Fwd", self.fast_forward, {}),
            ("Home", self.go_home, {}),
            ("End", self.go_end, {}),
        ]
        row2 = [
            ("Test Tone", self.test_tone, {"w": 100}),
            ("Save Mix", self.saveas, {"w": 100}),
            ("Save Stems", self.save_stems, {"w": 110}),
            ("+ Track", self.add_track, {}),
            ("- Track", self.delete_track, {}),
        ]
        row3 = [
            ("Import", self.import_clip, {"w": 100}),
            ("Delete", self.delete_selected, {}),
            ("Split", self.split_at_playhead, {"w": 90}),
            ("Zoom -", lambda: self.set_zoom(self.px_per_sec / 1.25), {"w": 80}),
            ("Zoom +", lambda: self.set_zoom(self.px_per_sec * 1.25), {"w": 80}),
            ("Snap", self.toggle_snap, {"toggle": True, "state": self.snap, "w": 80}),
        ]
        row4 = [
            ("New", self.new_project, {"w": 80}),
            ("Open", self.open_project, {"w": 80}),
            ("Save", self.save_project, {"w": 80}),
            ("Save As", self.save_project_as, {"w": 100}),
        ]

        self._topbar_buttons = (
            add_row(row1, 0) + add_row(row2, 1) + add_row(row3, 2) + add_row(row4, 3)
        )

        # Left panel inputs (labels rendered by panel)
        base_x = 20
        y0 = self.top_bar_h + 20
        self.bpm_input = TextInput(base_x, y0 + 18, 110, 30, "", str(self.bpm), True)
        self.time_sig_numerator = TextInput(base_x + 130, y0 + 18, 42, 30, "", "4", True)
        self.time_sig_denominator = TextInput(base_x + 188, y0 + 18, 42, 30, "", "4", True)

        # Sliders with added spacing
        self.sld_gain = Slider(
            base_x, y0 + 128, self.left_panel_w - 2 * base_x, "Clip Gain", 0.0, 2.0, 1.0
        )
        self.sld_speed = Slider(
            base_x, y0 + 188, self.left_panel_w - 2 * base_x, "Clip Speed", 0.25, 2.0, 1.0
        )

    def _reflow_ui_preserve(self):
        """Recreate UI widgets but keep current values/texts."""
        try:
            gain = self.sld_gain.val
            speed = self.sld_speed.val
            bpm_t = self.bpm_input.text
            ts_n = self.time_sig_numerator.text
            ts_d = self.time_sig_denominator.text
        except Exception:
            gain, speed, bpm_t, ts_n, ts_d = 1.0, 1.0, str(self.bpm), "4", "4"

        self._create_ui_elements()
        self.sld_gain.val = gain
        self.sld_speed.val = speed
        self.bpm_input.text = bpm_t
        self.time_sig_numerator.text = ts_n
        self.time_sig_denominator.text = ts_d

    # ---------- Layout / resize ----------

    def on_resize(self, w, h):
        self.WIDTH, self.HEIGHT = w, h
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.RESIZABLE)

        # Respect user-chosen panel width if set; clamp to sensible range
        max_panel = int(self.WIDTH * self.MAX_LEFT_PANEL_RATIO)
        if self.user_left_panel_w is None:
            self.left_panel_w = min(360, max(260, self.WIDTH // 4))
        else:
            self.left_panel_w = int(max(self.MIN_LEFT_PANEL, min(self.user_left_panel_w, max_panel)))

        self.timeline_origin_x = self.left_panel_w + self.track_header_w
        self._reflow_ui_preserve()

        total_px = self.total_track_pixels()
        vis_px = self.HEIGHT - self.top_bar_h - self.ruler_h - self.bottom_bar_h
        self.v_off_px = min(self.v_off_px, max(0, total_px - vis_px))

        total_sec = max(self.project_length(), self.visible_secs())
        vis_sec = self.visible_secs()
        self.h_off_sec = max(0.0, min(self.h_off_sec, total_sec - vis_sec))

    def _create_track(self):
        idx = len(self.tracks) + 1
        self.tracks.append(Track(f"Track {idx}"))

    def add_track(self):
        self._create_track()
        self.selected_track = len(self.tracks) - 1
        self.mark_project_modified()
        self.set_status(f"Added Track {self.selected_track + 1}")

    def delete_track(self):
        if len(self.tracks) <= 1:
            self.set_status("Cannot delete the last track.", ok=False)
            return
        if self.selected_track >= len(self.tracks):
            self.selected_track = len(self.tracks) - 1
        if self.selected_clip and self.selected_clip.track_index == self.selected_track:
            self.select_clip(None)
        if self.selected_track > 0:
            prev = self.tracks[self.selected_track - 1]
            for c in self.tracks[self.selected_track].clips:
                c.track_index = self.selected_track - 1
                prev.clips.append(c)
        del self.tracks[self.selected_track]
        if self.selected_track >= len(self.tracks):
            self.selected_track = len(self.tracks) - 1
        self.mark_project_modified()
        self.invalidate_render()
        self.set_status("Deleted track.")

    # ---------- Helpers ----------

    def set_status(self, msg, ok=True):
        self.status = msg
        self.status_col = COL_OK if ok else COL_ERR
        print(msg)

    def visible_secs(self) -> float:
        return max(0.5, (self.WIDTH - self.timeline_origin_x - self.right_bar_w) / self.px_per_sec)

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
        for b in self._topbar_buttons:
            if b.label == "Snap":
                b.state = self.snap
                break
        self.mark_project_modified()
        self.set_status(f"Snap {'ON' if self.snap else 'OFF'}")

    def select_clip(self, clip: Optional[Clip]):
        if self.selected_clip is not None:
            self.selected_clip.selected = False
        self.selected_clip = clip
        if clip is not None:
            clip.selected = True
            if self.drag_mode is None:
                self.sld_gain.val = clip.gain
                self.sld_speed.val = clip.speed

    # ---------- Import / edit ----------

    def import_clip(self):
        try:
            if sys.platform == "darwin":
                import subprocess
                osa = 'POSIX path of (choose file with prompt "Import Audio" of type {"public.audio"})'
                res = subprocess.run(["osascript", "-e", osa], capture_output=True, text=True)
                path = res.stdout.strip() if res.returncode == 0 else None
            else:
                from tkinter import filedialog, Tk
                root = Tk()
                root.withdraw()
                root.update()
                path = filedialog.askopenfilename(
                    title="Import Audio",
                    filetypes=[("Audio", "*.wav *.mp3 *.flac *.aiff *.aif *.ogg"), ("All files", "*.*")]
                )
                root.destroy()
        except Exception:
            path = None
        if not path:
            self.set_status("Import cancelled.", ok=True)
            return
        mx, my = pygame.mouse.get_pos()
        self._import_at_screen_pos(path, mx, my)

    def _ensure_track_for_y(self, y_px: int, create_if_needed: bool = False) -> int:
        top = self.top_bar_h + self.ruler_h
        if y_px < top:
            return self.selected_track if self.tracks else 0
        lane = int((y_px + self.v_off_px - top) // self.track_h)
        if lane < 0:
            return self.selected_track if self.tracks else 0
        if lane >= len(self.tracks):
            if create_if_needed:
                while len(self.tracks) <= lane:
                    self._create_track()
                self.mark_project_modified()
                self.set_status(f"Auto-created Track {lane + 1}")
            else:
                lane = max(0, len(self.tracks) - 1)
        return lane

    def _import_at_screen_pos(self, path: str, mx: int, my: int):
        try:
            y, _ = load_audio(path)
            if y.size == 0:
                self.set_status(f"Failed to load audio from {path}", ok=False)
                return
            name = os.path.basename(path)
            start_t = self.playhead
            ti = self._ensure_track_for_y(my, create_if_needed=True)
            c = Clip(y, name=name, start=start_t, speed=1.0, gain=1.0)
            c.track_index = ti
            self.tracks[ti].clips.append(c)
            self.select_clip(c)
            self.mark_project_modified()
            self.invalidate_render()
            self.set_status(f"Imported {name} → Track {ti + 1}")
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
        self.mark_project_modified()
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
            self.mark_project_modified()
            self.invalidate_render()
            self.set_status("Clip split.")
        else:
            self.set_status("Playhead not inside clip.", ok=False)

    # ---------- Transport ----------

    def toggle_loop(self):
        self.looping = not self.looping
        for b in self._topbar_buttons:
            if b.label == "Loop":
                b.state = self.looping
                break
        self.mark_project_modified()
        self.set_status(f"Looping {'ON' if self.looping else 'OFF'}")

    def rewind(self):
        self.playhead = max(0.0, self.playhead - 5.0)
        self.set_status("Rewound 5 seconds")

    def fast_forward(self):
        self.playhead = min(self.project_length(), self.playhead + 5.0)
        self.set_status("Fast forwarded 5 seconds")

    def go_home(self):
        self.playhead = 0.0
        self.set_status("Moved to start")

    def go_end(self):
        self.playhead = self.project_length()
        self.set_status("Moved to end")

    def pause(self):
        if self.playing:
            if self.paused:
                self.play_start_clock = time.perf_counter() - (self.playhead - self.play_start_pos)
                self.paused = False
                pygame.mixer.unpause()
                self.set_status("Resumed playback")
            else:
                self.paused = True
                pygame.mixer.pause()
                self.set_status("Playback paused")
        else:
            self.set_status("Not playing", ok=False)

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

                start_sample = int(c.in_pos * SR)
                end_sample = int(c.out_pos * SR)
                if start_sample >= end_sample:
                    continue

                seg = c.audio[start_sample:end_sample].copy()
                if seg.size == 0:
                    continue

                stretched = time_stretch(seg, c.speed)
                if stretched.size == 0:
                    continue

                apply_fades(stretched, c.fade_in, c.fade_out, SR)

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
                    stem[i0:i0 + take] += stretched[:take] * c.gain

            stems[ti] = stems[ti] * tr.volume

        mix = np.sum(stems, axis=0)

        if mix.size:
            peak = float(np.max(np.abs(mix)))
            if peak > 1.0:
                mix /= (peak + 1e-6)
            mix = np.tanh(mix * 1.2)
            peak2 = float(np.max(np.abs(mix)))
            target = 0.89
            if peak2 > 1e-6 and peak2 < target:
                mix *= (target / peak2)

        return mix, start_sec, stems

    def invalidate_render(self):
        self._stems = None
        self.mix_sound = None

    def _play_array(self, mono: np.ndarray, start_idx: int):
        if mono.size == 0 or start_idx >= mono.size:
            self.set_status("Nothing to play at playhead.", ok=False)
            return False
        chunk = mono[start_idx:]
        if not chunk.size or float(np.max(np.abs(chunk))) < 1e-6:
            self.set_status("Buffer is silent at playhead.", ok=False)
            return False
        stereo = to_stereo(chunk)
        stereo_i16 = to_int16_stereo(stereo)
        try:
            snd = pygame.sndarray.make_sound(stereo_i16)
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
        if self.playing and not self.paused:
            self.stop()
            return
        if self.paused:
            self.paused = False
            pygame.mixer.unpause()
            self.play_start_clock = time.perf_counter() - (self.playhead - self.play_start_pos)
            self.set_status("Resumed playback")
            return
        try:
            mix, _, stems = self.render_mix_with_stems(0.0)
            self._stems = stems
            i0 = int(self.playhead * SR)
            ok = self._play_array(mix, i0)
            if not ok:
                return
            self.play_start_pos = self.playhead
            self.play_start_clock = time.perf_counter()
            self.play_end_sec = len(mix) / SR
            self.playing = True
            self.paused = False
            self.set_status("Playing…", ok=True)
        except Exception as e:
            self.set_status(f"Play failed: {e}", ok=False)

    def stop(self):
        try:
            pygame.mixer.stop()
        except Exception:
            pass
        self.playing = False
        self.paused = False
        self.play_channel = None
        self.play_start_clock = None
        self.set_status("Stopped.", ok=True)

    def test_tone(self):
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

    # ---------- Export ----------

    def saveas(self):
        try:
            if sys.platform == "darwin":
                import subprocess
                osa = 'POSIX path of (choose file name with prompt "Save Mix As" default name "mix.wav")'
                res = subprocess.run(["osascript", "-e", osa], capture_output=True, text=True)
                path = res.stdout.strip() if res.returncode == 0 else None
                if path and not path.lower().endswith(".wav"):
                    path += ".wav"
            else:
                from tkinter import filedialog, Tk
                root = Tk()
                root.withdraw()
                root.update()
                path = filedialog.asksaveasfilename(
                    title="Save Mix As",
                    defaultextension=".wav",
                    filetypes=[("WAV", "*.wav")],
                    initialfile="mix.wav"
                )
                root.destroy()
        except Exception:
            path = None

        if not path:
            self.set_status("Save cancelled.", ok=True)
            return

        mix, _, _ = self.render_mix_with_stems(0.0)
        stereo = to_stereo(mix)
        try:
            import wave
            with wave.open(path, "wb") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(SR)
                wf.writeframes(to_int16_stereo(stereo).tobytes())
            self.set_status(f"Saved mix → {path}")
        except Exception as e:
            self.set_status(f"Save failed: {e}", ok=False)

    def save_stems(self):
        try:
            if sys.platform == "darwin":
                import subprocess
                osa = 'POSIX path of (choose folder with prompt "Choose folder for stems")'
                res = subprocess.run(["osascript", "-e", osa], capture_output=True, text=True)
                folder = res.stdout.strip() if res.returncode == 0 else None
            else:
                from tkinter import filedialog, Tk
                root = Tk()
                root.withdraw()
                root.update()
                folder = filedialog.askdirectory(title="Choose folder for stems")
                root.destroy()
        except Exception:
            folder = None

        if not folder:
            self.set_status("Stem export cancelled.", ok=True)
            return

        _, _, stems = self.render_mix_with_stems(0.0)
        try:
            import wave
            for i, (tr, stem) in enumerate(zip(self.tracks, stems)):
                st = to_stereo(stem)
                name = f"{i + 1:02d}_{tr.name.replace(' ', '_')}.wav"
                out = os.path.join(folder, name)
                with wave.open(out, "wb") as wf:
                    wf.setnchannels(2)
                    wf.setsampwidth(2)
                    wf.setframerate(SR)
                    wf.writeframes(to_int16_stereo(st).tobytes())
            self.set_status(f"Exported stems → {folder}")
        except Exception as e:
            self.set_status(f"Stem export failed: {e}", ok=False)

    # ---------- Meters ----------

    def update_meters(self):
        if not self.playing or self._stems is None:
            for tr in self.tracks:
                tr.meter_level = 0.0
            return
        i = int(self.playhead * SR)
        win = int(0.05 * SR)
        for tr, stem in zip(self.tracks, self._stems):
            if i >= len(stem):
                tr.meter_level = 0.0
                continue
            s = stem[i:i + win]
            tr.meter_level = float(
                min(1.0, max(0.0, np.sqrt(np.mean((s * tr.volume) ** 2)) * 2.0))
            )

    # ---------- Drawing ----------

    def draw(self):
        self.screen.fill((20, 20, 26))
        self.draw_topbar()
        self.draw_left_panel()
        self.draw_ruler_and_grid()
        self.draw_tracks()
        self.draw_playhead()
        self.draw_scrollbars()

        proj_name = os.path.basename(self.project_path) if self.project_path else "Untitled"
        modified = "*" if self.project_modified else ""
        proj_text = f"{proj_name}{modified}"
        txt = FONT_SM.render(proj_text, True, COL_TXT)
        self.screen.blit(txt, (self.WIDTH - 150, self.HEIGHT - self.bottom_bar_h + 4))

        txt = FONT_SM.render(self.status, True, self.status_col)
        self.screen.blit(txt, (10, self.HEIGHT - self.bottom_bar_h + 4))

    def draw_topbar(self):
        pygame.draw.rect(self.screen, COL_PANEL, (0, 0, self.WIDTH, self.top_bar_h))
        pygame.draw.line(self.screen, (60, 60, 80), (0, self.top_bar_h), (self.WIDTH, self.top_bar_h), 2)
        mouse = pygame.mouse.get_pos()
        for b in self._topbar_buttons:
            b.update(mouse)
            b.draw(self.screen)

    def draw_left_panel(self):
        # Left panel
        panel_rect = pygame.Rect(0, self.top_bar_h, self.left_panel_w, self.HEIGHT - self.top_bar_h)
        pygame.draw.rect(self.screen, COL_PANEL, panel_rect)

        # Splitter (draggable)
        split_x = self.left_panel_w
        splitter_rect = pygame.Rect(
            split_x - self.SPLIT_HIT_PAD // 2,
            self.top_bar_h,
            self.SPLIT_HIT_PAD,
            self.HEIGHT - self.top_bar_h
        )
        hover = splitter_rect.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(
            self.screen, (60, 60, 80),
            (split_x - 1, self.top_bar_h, 2, self.HEIGHT - self.top_bar_h)
        )
        pygame.draw.rect(
            self.screen,
            COL_SCROLL_KNOB if (hover or self.splitter_drag) else (70, 70, 90),
            splitter_rect,
            0
        )

        # Divider toward track header
        pygame.draw.line(
            self.screen,
            (60, 60, 80),
            (self.left_panel_w, self.top_bar_h),
            (self.left_panel_w, self.HEIGHT),
            2
        )

        # Panel content
        x = 20
        y = self.top_bar_h + 10

        self.screen.blit(FONT_MD.render("Project Settings", True, COL_TXT), (x, y))
        y += 28

        self.screen.blit(FONT_SM.render("BPM", True, COL_TXT), (x, y))
        self.bpm_input.rect.topleft = (x, y + 18)
        self.bpm_input.draw(self.screen)

        self.screen.blit(FONT_SM.render("Time Signature", True, COL_TXT), (x + 130, y))
        self.time_sig_numerator.rect.topleft = (x + 130, y + 18)
        self.time_sig_denominator.rect.topleft = (x + 188, y + 18)
        self.time_sig_numerator.draw(self.screen)
        self.time_sig_denominator.draw(self.screen)
        div = FONT_SM.render("/", True, COL_TXT)
        self.screen.blit(div, (x + 176, y + 24 - div.get_height() // 2))

        # extra padding before Clip Controls
        y += 84

        self.screen.blit(FONT_MD.render("Clip Controls", True, COL_TXT), (x, y))
        y += 20

        self.sld_gain.rect.topleft = (x, y + 10)
        self.sld_gain.rect.width = self.left_panel_w - 2 * x
        self.sld_gain.draw(self.screen)
        y += 60

        self.sld_speed.rect.topleft = (x, y + 6)
        self.sld_speed.rect.width = self.left_panel_w - 2 * x
        self.sld_speed.draw(self.screen)
        y += 52

        if self.selected_clip:
            self.screen.blit(FONT_MD.render("Selected Clip", True, COL_TXT), (x, y))
            y += 6
            info_rect = pygame.Rect(x, y + 20, self.left_panel_w - 40, 120)
            pygame.draw.rect(self.screen, COL_BPM_BG, info_rect, border_radius=6)
            pygame.draw.rect(self.screen, (70, 70, 90), info_rect, 2, border_radius=6)
            self.screen.blit(FONT_SM.render(f"Name: {self.selected_clip.name}", True, COL_TXT),
                             (info_rect.x + 10, info_rect.y + 10))
            self.screen.blit(FONT_SM.render(f"Position: {self.selected_clip.start:.2f}s", True, COL_TXT),
                             (info_rect.x + 10, info_rect.y + 30))
            self.screen.blit(FONT_SM.render(f"Length: {self.selected_clip.eff_len_sec:.2f}s", True, COL_TXT),
                             (info_rect.x + 10, info_rect.y + 50))
            self.screen.blit(FONT_SM.render(f"Gain: {self.selected_clip.gain:.2f}", True, COL_TXT),
                             (info_rect.x + 10, info_rect.y + 70))
            self.screen.blit(FONT_SM.render(f"Speed: {self.selected_clip.speed:.2f}", True, COL_TXT),
                             (info_rect.x + 10, info_rect.y + 90))

    def draw_ruler_and_grid(self):
        y0 = self.top_bar_h
        pygame.draw.rect(self.screen, (28, 28, 38),
                         (self.left_panel_w, y0, self.WIDTH - self.left_panel_w, self.ruler_h))

        left_t = self.h_off_sec
        right_t = self.h_off_sec + self.visible_secs()
        beat = 60.0 / max(1e-6, self.bpm)
        bar = beat * 4
        first_bar = math.floor(left_t / bar)
        t = first_bar * bar
        while t < right_t + bar:
            x = self.time_to_x(t)
            pygame.draw.line(self.screen, COL_GRID_BAR, (x, y0), (x, self.HEIGHT - self.bottom_bar_h), 2)
            for b in range(1, 4):
                tb = t + b * beat
                if left_t <= tb <= right_t:
                    xb = self.time_to_x(tb)
                    pygame.draw.line(self.screen, COL_GRID, (xb, y0 + self.ruler_h // 2),
                                     (xb, self.HEIGHT - self.bottom_bar_h), 1)
            if x >= self.timeline_origin_x - 30:
                label = FONT_SM.render(str(int(t / bar)), True, COL_TXT)
                self.screen.blit(label, (x - 4, y0 + 4))
            t += bar

        # header column
        pygame.draw.rect(self.screen, COL_PANEL,
                         (self.left_panel_w, self.top_bar_h + self.ruler_h,
                          self.track_header_w,
                          self.HEIGHT - self.top_bar_h - self.ruler_h))

    def draw_tracks(self):
        for i, tr in enumerate(self.tracks):
            y = self.track_y(i)
            if y + self.track_h < self.top_bar_h + self.ruler_h or y > self.HEIGHT - self.bottom_bar_h:
                continue

            hdr = pygame.Rect(self.left_panel_w, y, self.track_header_w, self.track_h - 4)
            pygame.draw.rect(self.screen, (30, 30, 40), hdr)
            pygame.draw.rect(self.screen, (55, 55, 70), hdr, 1)

            title_col = COL_ACC if i == self.selected_track else COL_TXT
            self.screen.blit(FONT_MD.render(tr.name, True, title_col), (self.left_panel_w + 10, y + 6))

            tr._mute_rect = pygame.Rect(self.left_panel_w + 12, y + 30, 24, 18)
            pygame.draw.rect(self.screen, (210, 80, 80) if tr.mute else (70, 70, 80), tr._mute_rect, border_radius=4)
            self.screen.blit(FONT_SM.render("M", True, COL_TXT), tr._mute_rect.move(7, 1))

            tr._solo_rect = pygame.Rect(self.left_panel_w + 42, y + 30, 24, 18)
            pygame.draw.rect(self.screen, (240, 200, 90) if tr.solo else (70, 70, 80), tr._solo_rect, border_radius=4)
            self.screen.blit(FONT_SM.render("S", True, COL_TXT), tr._solo_rect.move(7, 1))

            tr._delete_rect = pygame.Rect(self.left_panel_w + 72, y + 30, 24, 18)
            pygame.draw.rect(self.screen, (220, 80, 80), tr._delete_rect, border_radius=4)
            self.screen.blit(FONT_SM.render("X", True, COL_TXT), tr._delete_rect.move(7, 1))

            fx, fw = self.left_panel_w + 110, self.track_header_w - 130
            tr._fader_rect = pygame.Rect(fx, y + 30, fw, 6)
            pygame.draw.rect(self.screen, (60, 60, 80), tr._fader_rect, border_radius=3)
            knob_x = int(fx + tr.volume * (fw - 1))
            pygame.draw.circle(self.screen, COL_ACC2, (knob_x, y + 33), 7)

            meter = pygame.Rect(self.left_panel_w + self.track_header_w - 16, y + 6, 8, self.track_h - 20)
            pygame.draw.rect(self.screen, COL_METER_BG, meter)
            m_h = int((self.track_h - 20) * max(0.0, min(1.0, tr.meter_level)))
            if m_h > 0:
                pygame.draw.rect(self.screen, COL_METER, (meter.x, meter.bottom - m_h, meter.w, m_h))

            lane = pygame.Rect(self.timeline_origin_x, y,
                               self.WIDTH - self.timeline_origin_x - self.right_bar_w,
                               self.track_h - 4)
            col = (24, 24, 30) if i % 2 == 0 else (22, 22, 28)
            pygame.draw.rect(self.screen, col, lane)

            for c in tr.clips:
                left, right = c.bounds()
                if right < self.h_off_sec or left > self.h_off_sec + self.visible_secs():
                    continue
                x1 = self.time_to_x(left)
                x2 = self.time_to_x(right)
                rect = pygame.Rect(x1, y + 6, max(32, x2 - x1), self.track_h - 16)
                c.rect = rect
                pygame.draw.rect(self.screen, COL_CLIP_SEL if c.selected else COL_CLIP, rect, border_radius=6)
                pygame.draw.rect(self.screen, (50, 50, 70), rect, 2, border_radius=6)
                label = FONT_SM.render(f"{c.name}  x{c.speed:.2f}  g{c.gain:.2f}", True, COL_TXT)
                self.screen.blit(label, (rect.x + 6, rect.y + 4))
                fh = 10
                pygame.draw.polygon(self.screen, COL_FADE, [(rect.x, rect.y), (rect.x + fh, rect.y), (rect.x, rect.y + fh)])
                pygame.draw.polygon(self.screen, COL_FADE, [(rect.right, rect.y), (rect.right - fh, rect.y), (rect.right, rect.y + fh)])

    def draw_playhead(self):
        x = self.time_to_x(self.playhead)
        pygame.draw.line(self.screen, (240, 80, 80),
                         (x, self.top_bar_h), (x, self.HEIGHT - self.bottom_bar_h), 2)

    def draw_scrollbars(self):
        # Horizontal track area
        y = self.HEIGHT - self.bottom_bar_h - self.scrollbar_h
        w = self.WIDTH - self.timeline_origin_x - self.right_bar_w
        self.h_bar_rect = pygame.Rect(self.timeline_origin_x, y, max(0, w), self.scrollbar_h)
        pygame.draw.rect(self.screen, COL_SCROLL, self.h_bar_rect)
        pygame.draw.rect(self.screen, (70, 70, 90), self.h_bar_rect, 1)

        total = max(self.project_length(), self.visible_secs())
        vis = self.visible_secs()
        if total > 0 and w > 0:
            frac = max(0.06, min(1.0, vis / total))
            span_px = int(w * frac)
            max_off = max(0.0, total - vis)
            off_frac = 0.0 if max_off <= 0 else (self.h_off_sec / max_off) * (1 - frac)
            knob_x = int(self.timeline_origin_x + off_frac * w)
            self.h_scroll_rect = pygame.Rect(knob_x, y + 2, span_px, self.scrollbar_h - 4)
            pygame.draw.rect(self.screen, COL_SCROLL_KNOB, self.h_scroll_rect, border_radius=4)
        else:
            self.h_scroll_rect = pygame.Rect(self.timeline_origin_x, y + 2, w, self.scrollbar_h - 4)
            pygame.draw.rect(self.screen, COL_SCROLL_KNOB, self.h_scroll_rect, border_radius=4)

        # Vertical track area
        x = self.WIDTH - self.right_bar_w
        h = self.HEIGHT - self.top_bar_h - self.ruler_h - self.bottom_bar_h
        self.v_bar_rect = pygame.Rect(x, self.top_bar_h + self.ruler_h, self.scrollbar_w, max(0, h))
        pygame.draw.rect(self.screen, COL_SCROLL, self.v_bar_rect)
        pygame.draw.rect(self.screen, (70, 70, 90), self.v_bar_rect, 1)

        total_px = self.total_track_pixels()
        vis_px = h
        if total_px > 0 and h > 0:
            frac = max(0.08, min(1.0, vis_px / total_px))
            span = int(h * frac)
            max_off = max(0, total_px - vis_px)
            off_frac = 0.0 if max_off <= 0 else (self.v_off_px / max_off) * (1 - frac)
            knob_y = int(self.top_bar_h + self.ruler_h + off_frac * h)
            self.v_scroll_rect = pygame.Rect(x + 2, knob_y + 2, self.scrollbar_w - 4, max(8, span - 4))
            pygame.draw.rect(self.screen, COL_SCROLL_KNOB, self.v_scroll_rect, border_radius=4)
        else:
            self.v_scroll_rect = pygame.Rect(x + 2, self.top_bar_h + self.ruler_h + 2, self.scrollbar_w - 4, max(8, h - 4))
            pygame.draw.rect(self.screen, COL_SCROLL_KNOB, self.v_scroll_rect, border_radius=4)

    # ---------- Interaction ----------

    def wheel(self, e):
        mods = pygame.key.get_mods()
        if mods & (pygame.KMOD_CTRL | pygame.KMOD_META):
            mx, _ = pygame.mouse.get_pos()
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
                vis_px = self.HEIGHT - self.top_bar_h - self.ruler_h - self.bottom_bar_h
                max_off = max(0, total_px - vis_px)
                self.v_off_px = int(max(0, min(max_off, self.v_off_px - e.y * 60)))

    def key(self, e):
        mods = pygame.key.get_mods()
        ctrl_or_cmd = (mods & pygame.KMOD_CTRL) or (mods & pygame.KMOD_META)

        if e.key == pygame.K_SPACE:
            self.stop() if self.playing else self.play()
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
        elif e.key == pygame.K_LEFT:
            self.playhead = max(0.0, self.playhead - 1.0)
        elif e.key == pygame.K_RIGHT:
            self.playhead = min(self.project_length(), self.playhead + 1.0)
        elif e.key == pygame.K_z and self.selected_clip:
            l, r = self.selected_clip.bounds()
            mid = (l + r) / 2
            span = max(0.25, r - l)
            self.set_zoom(max(60, min(400, (self.WIDTH - self.timeline_origin_x) / span)), anchor_time=mid)
            self.h_off_sec = max(0.0, l - 0.25 * span)
        elif ctrl_or_cmd and (mods & pygame.KMOD_SHIFT) and e.key == pygame.K_s:
            self.save_project_as()
        elif ctrl_or_cmd and e.key == pygame.K_s:
            self.save_project()
        elif ctrl_or_cmd and e.key == pygame.K_m:
            self.saveas()
        elif ctrl_or_cmd and e.key == pygame.K_e:
            self.save_stems()
        elif ctrl_or_cmd and e.key == pygame.K_l:
            self.toggle_loop()
        elif ctrl_or_cmd and e.key == pygame.K_n:
            self.new_project()
        elif ctrl_or_cmd and e.key == pygame.K_o:
            self.open_project()

    def on_fader_drag(self, mx):
        tr = self.tracks[self.drag_fader_track]
        rel = (mx - tr._fader_rect.x) / max(1, tr._fader_rect.w - 1)
        tr.volume = max(0.0, min(1.5, rel))
        self.invalidate_render()
        self.mark_project_modified()

    def mouse_down(self, e):
        mx, my = e.pos
        self.drag_last_mouse = (mx, my)

        # Splitter drag start?
        if abs(mx - self.left_panel_w) <= self.SPLIT_HIT_PAD and my >= self.top_bar_h:
            self.splitter_drag = True
            self._splitter_grab_dx = mx - self.left_panel_w
            return

        # Topbar buttons
        for button in self._topbar_buttons:
            if button.rect.collidepoint(mx, my):
                button.handle_click()
                return

        # Inputs
        for ti in (self.bpm_input, self.time_sig_numerator, self.time_sig_denominator):
            if ti.rect.collidepoint(mx, my):
                ti.active = True
                return

        # Scrollbar hit-tests (bars jump; knobs drag)
        if self.h_bar_rect.collidepoint(mx, my):
            if self.h_scroll_rect.collidepoint(mx, my):
                self._drag_scroll_h = True
            else:
                # jump to click location
                w = self.h_bar_rect.w
                if w > 0:
                    frac = (mx - self.h_bar_rect.x) / w
                    total = max(self.project_length(), self.visible_secs())
                    vis = self.visible_secs()
                    max_off = max(0.0, total - vis)
                    self.h_off_sec = max(0.0, min(max_off, frac * max_off))
            return

        if self.v_bar_rect.collidepoint(mx, my):
            if self.v_scroll_rect.collidepoint(mx, my):
                self._drag_scroll_v = True
            else:
                h = self.v_bar_rect.h
                if h > 0:
                    frac = (my - self.v_bar_rect.y) / h
                    total_px = self.total_track_pixels()
                    vis_px = self.HEIGHT - self.top_bar_h - self.ruler_h - self.bottom_bar_h
                    max_off = max(0, total_px - vis_px)
                    self.v_off_px = int(max(0, min(max_off, frac * max_off)))
            return

        # Track header region
        top_tracks = self.top_bar_h + self.ruler_h
        if my >= top_tracks and self.left_panel_w <= mx < self.timeline_origin_x:
            ti = self._ensure_track_for_y(my, create_if_needed=False)
            self.selected_track = ti
            tr = self.tracks[ti]
            if getattr(tr, "_fader_rect", None) and tr._fader_rect.collidepoint(mx, my):
                self.drag_mode = "fader"
                self.drag_fader_track = ti
                return
            if getattr(tr, "_mute_rect", None) and tr._mute_rect.collidepoint(mx, my):
                tr.mute = not tr.mute
                self.invalidate_render()
                self.mark_project_modified()
                return
            if getattr(tr, "_solo_rect", None) and tr._solo_rect.collidepoint(mx, my):
                tr.solo = not tr.solo
                self.invalidate_render()
                self.mark_project_modified()
                return
            if getattr(tr, "_delete_rect", None) and tr._delete_rect.collidepoint(mx, my):
                self.delete_track()
                return
            return

        # Ruler scrub
        if self.top_bar_h <= my <= self.top_bar_h + self.ruler_h and mx >= self.timeline_origin_x:
            self.playhead = max(0.0, self.x_to_time(mx))
            self.drag_mode = "scrub"
            return

        # Right-button pan
        if e.button == 3:
            self.drag_mode = "pan"
            return

        # Timeline lanes
        if mx >= self.timeline_origin_x and my >= self.top_bar_h + self.ruler_h:
            # hit clips first
            for ti in range(len(self.tracks)):
                tr = self.tracks[ti]
                for c in reversed(tr.clips):
                    if c.rect and c.rect.collidepoint(mx, my):
                        self.select_clip(c)
                        if mx - c.rect.x < 8:
                            self.drag_mode = "trim_l"
                        elif c.rect.right - mx < 8:
                            self.drag_mode = "trim_r"
                        elif my - c.rect.y < 14 and mx - c.rect.x < 14:
                            self.drag_mode = "fade_l"
                        elif my - c.rect.y < 14 and c.rect.right - mx < 14:
                            self.drag_mode = "fade_r"
                        else:
                            self.drag_mode = "move"
                            self.drag_offset_time = self.x_to_time(mx) - c.start
                        self.drag_clip_ref = c
                        return
            # empty lane: set playhead; clamp to existing tracks only
            self.playhead = self.maybe_snap(self.x_to_time(mx))
            self.selected_track = self._ensure_track_for_y(my, create_if_needed=False)
            self.select_clip(None)

    def mouse_up(self, e):
        if self.drag_mode in ("move", "trim_l", "trim_r", "fade_l", "fade_r"):
            self.mark_project_modified()
        self.drag_mode = None
        self.drag_clip_ref = None
        self.drag_fader_track = None
        self._drag_scroll_h = False
        self._drag_scroll_v = False

        if self.splitter_drag:
            self.splitter_drag = False
            self.user_left_panel_w = self.left_panel_w  # remember user width

    def mouse_motion(self, e):
        mx, my = e.pos
        dx = mx - self.drag_last_mouse[0]
        dy = my - self.drag_last_mouse[1]
        self.drag_last_mouse = (mx, my)

        # Splitter drag
        if self.splitter_drag:
            max_panel = int(self.WIDTH * self.MAX_LEFT_PANEL_RATIO)
            new_w = int(mx - self._splitter_grab_dx)
            new_w = max(self.MIN_LEFT_PANEL, min(max_panel, new_w))
            if new_w != self.left_panel_w:
                self.left_panel_w = new_w
                self.timeline_origin_x = self.left_panel_w + self.track_header_w
                self._reflow_ui_preserve()
            return

        if self.drag_mode == "scrub":
            self.playhead = max(0.0, self.x_to_time(mx))
            return

        if self._drag_scroll_h:
            total = max(self.project_length(), self.visible_secs())
            vis = self.visible_secs()
            max_off = max(0.0, total - vis)
            w = max(1, self.h_bar_rect.w)
            self.h_off_sec = max(0.0, min(max_off, self.h_off_sec + (dx / w) * max_off))
            return

        if self._drag_scroll_v:
            total_px = self.total_track_pixels()
            vis_px = self.HEIGHT - self.top_bar_h - self.ruler_h - self.bottom_bar_h
            max_off = max(0, total_px - vis_px)
            h = max(1, self.v_bar_rect.h)
            self.v_off_px = int(max(0, min(max_off, self.v_off_px + (dy / h) * max_off)))
            return

        if self.drag_mode == "pan":
            self.h_off_sec = max(0.0, self.h_off_sec - dx / self.px_per_sec)
            total_px = self.total_track_pixels()
            vis_px = self.HEIGHT - self.top_bar_h - self.ruler_h - self.bottom_bar_h
            max_off = max(0, total_px - vis_px)
            self.v_off_px = int(max(0, min(max_off, self.v_off_px - dy)))
            return

        c = self.drag_clip_ref
        if c is None:
            return

        if self.drag_mode == "move":
            new_start = self.x_to_time(mx) - self.drag_offset_time
            c.start = self.maybe_snap(new_start)
            c.track_index = self._ensure_track_for_y(my, create_if_needed=False)  # clamp, don't create
            self.invalidate_render()
            return

        if self.drag_mode == "trim_l":
            t = self.x_to_time(mx)
            left, right = c.bounds()
            t = min(max(0.0, t), right - 0.01)
            if self.snap:
                t = self.maybe_snap(t)
            delta = (t - c.start) * c.speed
            c.in_pos += delta
            c.in_pos = max(0.0, min(c.in_pos, c.out_pos - 1e-3))
            c.start = t
            self.invalidate_render()
            return

        if self.drag_mode == "trim_r":
            t = max(c.start + 0.01, self.x_to_time(mx))
            if self.snap:
                t = self.maybe_snap(t)
            consumed = (t - c.start) * c.speed
            c.out_pos = max(c.in_pos + 1e-3, c.in_pos + consumed)
            self.invalidate_render()
            return

        if self.drag_mode == "fade_l":
            if c.rect:
                px = max(0, min(c.rect.w, mx - c.rect.x))
                c.fade_in = (px / self.px_per_sec)
                self.mark_project_modified()
            return

        if self.drag_mode == "fade_r":
            if c.rect:
                px = max(0, min(c.rect.w, c.rect.right - mx))
                c.fade_out = (px / self.px_per_sec)
                self.mark_project_modified()
            return

    # ---------- Update / events ----------

    def handle_event(self, e):
        if e.type == pygame.MOUSEWHEEL:
            self.wheel(e)
        elif e.type == pygame.KEYDOWN:
            self.key(e)
        elif e.type == pygame.MOUSEBUTTONDOWN:
            self.mouse_down(e)
        elif e.type == pygame.MOUSEBUTTONUP:
            self.mouse_up(e)
        elif e.type == pygame.MOUSEMOTION:
            self.mouse_motion(e)
        elif e.type == pygame.VIDEORESIZE:
            self.on_resize(e.w, e.h)

    def update(self, dt_ms: int, events):
        self.bpm_input.update(events, dt_ms)
        self.time_sig_numerator.update(events, dt_ms)
        self.time_sig_denominator.update(events, dt_ms)

        try:
            new_bpm = float(self.bpm_input.text)
        except Exception:
            new_bpm = self.bpm
        if abs(new_bpm - self.bpm) > 1e-6:
            self.bpm = max(20.0, min(300.0, new_bpm))

        mouse = pygame.mouse.get_pos()
        pressed = pygame.mouse.get_pressed()
        self.sld_gain.update(mouse, pressed)
        self.sld_speed.update(mouse, pressed)

        if self.selected_clip and self.drag_mode is None:
            if abs(self.selected_clip.gain - self.sld_gain.val) > 1e-6:
                self.selected_clip.gain = self.sld_gain.val
                self.invalidate_render()
                self.mark_project_modified()
            if abs(self.selected_clip.speed - self.sld_speed.val) > 1e-6:
                self.selected_clip.speed = self.sld_speed.val
                self.invalidate_render()
                self.mark_project_modified()

        if self.playing and not self.paused and self.play_start_clock is not None:
            t = time.perf_counter() - self.play_start_clock
            self.playhead = self.play_start_pos + t
            if self.play_end_sec is not None and self.playhead >= self.play_end_sec:
                if self.looping:
                    self.playhead = 0.0
                    self.play_start_pos = 0.0
                    self.play_start_clock = time.perf_counter()
                else:
                    self.stop()

        self.update_meters()

    # ---------- Project I/O ----------

    def mark_project_modified(self):
        self.project_modified = True

    def new_project(self):
        self.tracks = []
        self._create_track()
        self._create_track()
        self.selected_track = 0
        self.select_clip(None)
        self.playhead = 0.0
        self.bpm = 120.0
        self.subdiv = 4
        self.invalidate_render()
        self.project_path = None
        self.project_modified = False
        self.set_status("New project created.")

    def _serialize_project(self) -> Dict:
        data = {"bpm": self.bpm, "subdiv": self.subdiv, "tracks": []}
        for tr in self.tracks:
            tdict = tr.to_dict()
            clips_blob = []
            for c in tr.clips:
                cd = c.to_dict()
                cd["audio"] = c.audio
                clips_blob.append(cd)
            tdict["clips"] = clips_blob
            data["tracks"].append(tdict)
        return data

    def _deserialize_project(self, data: Dict):
        self.tracks = []
        self.bpm = float(data.get("bpm", 120.0))
        self.subdiv = int(data.get("subdiv", 4))
        for tdict in data.get("tracks", []):
            tr = Track.from_dict(tdict)
            for cd in tdict.get("clips", []):
                audio = cd.get("audio", np.array([], dtype=np.float32))
                clip = Clip.from_dict(cd, audio=audio)
                tr.clips.append(clip)
            self.tracks.append(tr)
        self.selected_track = 0 if self.tracks else -1
        self.select_clip(None)
        self.invalidate_render()
        self.project_modified = False

    def save_project(self):
        if not self.project_path:
            return self.save_project_as()
        try:
            with open(self.project_path, "wb") as f:
                pickle.dump(self._serialize_project(), f, protocol=pickle.HIGHEST_PROTOCOL)
            self.project_modified = False
            self.set_status(f"Project saved → {self.project_path}")
        except Exception as e:
            self.set_status(f"Save project failed: {e}", ok=False)

    def save_project_as(self):
        path = _choose_save_project(os.path.basename(self.project_path) if self.project_path else "project.pdaw")
        if not path:
            self.set_status("Save As cancelled.", ok=True)
            return
        self.project_path = path
        self.save_project()

    def open_project(self):
        if self.project_path and os.path.isfile(self.project_path):
            path = self.project_path
        else:
            path = _choose_open_project()
            if not path:
                self.set_status("Open cancelled.", ok=True)
                return
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._deserialize_project(data)
            self.project_path = path
            self.set_status(f"Project opened ← {path}")
        except Exception as e:
            self.set_status(f"Open project failed: {e}", ok=False)
