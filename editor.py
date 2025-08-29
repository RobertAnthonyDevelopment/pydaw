import os
import time
import math
import pickle
import pygame
import numpy as np
from typing import List, Optional, Dict, Tuple

from audio_utils import load_audio, time_stretch, to_stereo, to_int16_stereo, apply_fades, SR
from file_dialogs import choose_open_file, choose_save_file, choose_folder
from models import Clip, Track
from ui_components import *  # Button, Slider, TextInput, colors/fonts

class SimpleEditor:
    def __init__(self):
        # Initialize pygame
        pygame.init()
        pygame.sndarray.use_arraytype("numpy")
        
        # Screen setup
        self.WIDTH, self.HEIGHT = 1280, 860
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("PyDAW - Python Digital Audio Workstation")
        
        # Initialize mixer
        self._init_mixer()
        
        # Layout constants
        self.timeline_origin_x = 300
        self.track_h = 100
        self.ruler_h = 36
        self.top_bar_h = 180
        self.bottom_bar_h = 24
        self.right_bar_w = 16
        self.left_panel_w = 280

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
        self.paused = False
        self.looping = False
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
        
        # Project file management
        self.project_path: Optional[str] = None
        self.project_modified = False
        self.audio_files: Dict[str, np.ndarray] = {}  # Cache for loaded audio files
        
        # UI elements
        self._create_ui_elements()
        
        self.status = "Ready."
        self.status_col = COL_TXT

    def _init_mixer(self):
        """Initialize the pygame mixer with appropriate settings"""
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
                import sys
                sys.exit(1)

    def _create_ui_elements(self):
        """Create all UI elements with proper positioning"""
        # Transport controls (top row)
        x, y = 10, 10
        self.btn_play = Button(x, y, 60, 30, "Play", self.play); x += 70
        self.btn_pause = Button(x, y, 60, 30, "Pause", self.pause); x += 70
        self.btn_stop = Button(x, y, 60, 30, "Stop", self.stop); x += 70
        self.btn_loop = Button(x, y, 60, 30, "Loop", self.toggle_loop, toggle=True, state=self.looping); x += 70
        self.btn_rewind = Button(x, y, 60, 30, "Rewind", self.rewind); x += 70
        self.btn_ff = Button(x, y, 60, 30, "Fast Fwd", self.fast_forward); x += 70
        self.btn_home = Button(x, y, 60, 30, "Home", self.go_home); x += 70
        self.btn_end = Button(x, y, 60, 30, "End", self.go_end)
        
        # File operations (second row)
        x, y = 10, 50
        self.btn_tone = Button(x, y, 80, 30, "Test Tone", self.test_tone); x += 90
        self.btn_save = Button(x, y, 100, 30, "Save Mix", self.saveas); x += 110
        self.btn_save_stems = Button(x, y, 120, 30, "Save Stems", self.save_stems); x += 130
        self.btn_add_track = Button(x, y, 90, 30, "+ Track", self.add_track); x += 100
        self.btn_del_track = Button(x, y, 90, 30, "- Track", self.delete_track, color=COL_DELETE)
        
        # Editing controls (third row)
        x, y = 10, 90
        self.btn_import = Button(x, y, 100, 30, "Import", self.import_clip); x += 110
        self.btn_delete = Button(x, y, 80, 30, "Delete", self.delete_selected, color=COL_DELETE); x += 90
        self.btn_split = Button(x, y, 100, 30, "Split", self.split_at_playhead); x += 110
        self.btn_zoom_in = Button(x, y, 50, 30, "+", lambda: self.set_zoom(self.px_per_sec * 1.25)); x += 60
        self.btn_zoom_out = Button(x, y, 50, 30, "-", lambda: self.set_zoom(self.px_per_sec / 1.25)); x += 60
        self.btn_snap = Button(x, y, 70, 30, "Snap", self.toggle_snap, toggle=True, state=self.snap)
        
        # Project controls (fourth row)
        x, y = 10, 130
        self.btn_new = Button(x, y, 70, 30, "New", self.new_project); x += 80
        self.btn_open = Button(x, y, 70, 30, "Open", self.open_project); x += 80
        self.btn_save_proj = Button(x, y, 70, 30, "Save", self.save_project); x += 80
        self.btn_save_proj_as = Button(x, y, 90, 30, "Save As", self.save_project_as)

        # Text inputs - positioned in left panel
        self.bpm_input = TextInput(20, self.top_bar_h + 20, 100, 30, "BPM", str(self.bpm), True)
        self.time_sig_numerator = TextInput(130, self.top_bar_h + 20, 40, 30, "", "4", True)
        self.time_sig_denominator = TextInput(180, self.top_bar_h + 20, 40, 30, "", "4", True)

        # Sliders - positioned in left panel with proper spacing
        self.sld_gain = Slider(20, self.top_bar_h + 70, self.left_panel_w - 40, "Clip Gain", 0.0, 2.0, 1.0)
        self.sld_speed = Slider(20, self.top_bar_h + 110, self.left_panel_w - 40, "Clip Speed", 0.5, 1.5, 1.0)

    def on_resize(self, w, h):
        """Handle window resize events"""
        self.WIDTH, self.HEIGHT = w, h
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.RESIZABLE)
        
        # Adjust layout for new window size
        self.left_panel_w = min(300, self.WIDTH // 4)
        self.timeline_origin_x = self.left_panel_w + 10
        
        # Recreate UI elements to adjust to new size
        self._create_ui_elements()
        
        # Ensure view stays within bounds
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
            
        # If we're deleting a track with the selected clip, deselect it
        if self.selected_clip and self.selected_clip.track_index == self.selected_track:
            self.select_clip(None)
            
        # Move clips from the deleted track to the previous track if possible
        if self.selected_track > 0:
            prev_track = self.tracks[self.selected_track - 1]
            for clip in self.tracks[self.selected_track].clips:
                clip.track_index = self.selected_track - 1
                prev_track.clips.append(clip)
        
        # Remove the track
        del self.tracks[self.selected_track]
        
        # Adjust selected track index
        if self.selected_track >= len(self.tracks):
            self.selected_track = len(self.tracks) - 1
            
        self.mark_project_modified()
        self.invalidate_render()
        self.set_status(f"Deleted Track {self.selected_track + 1}")

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
            self.mark_project_modified()
            self.set_status(f"Auto-created Track {int(lane)+1}")
        return int(lane)

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
        self.btn_snap.state = self.snap
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

    def import_clip(self):
        path = choose_open_file("Choose audio")
        if not path:
            self.set_status("Import cancelled.", ok=True)
            return
        mx, my = pygame.mouse.get_pos()
        self._import_at_screen_pos(path, mx, my)

    def _import_at_screen_pos(self, path: str, mx: int, my: int):
        try:
            # Check if we already have this audio file loaded
            audio_key = os.path.basename(path)
            if audio_key in self.audio_files:
                y = self.audio_files[audio_key]
            else:
                y, _ = load_audio(path)
                if y.size == 0:
                    self.set_status(f"Failed to load audio from {path}", ok=False)
                    return
                # Cache the audio data
                self.audio_files[audio_key] = y
                
            name = os.path.basename(path)
            start_t = self.playhead
            ti = self._ensure_track_for_y(my)
            c = Clip(y, name=name, start=start_t, speed=1.0, gain=1.0)
            c.track_index = ti
            self.tracks[ti].clips.append(c)
            self.select_clip(c)
            self.mark_project_modified()
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

    def toggle_loop(self):
        self.looping = not self.looping
        self.btn_loop.state = self.looping
        self.mark_project_modified()
        self.set_status(f"Looping {'ON' if self.looping else 'OFF'}")

    def rewind(self):
        self.playhead = max(0.0, self.playhead - 5.0)  # Rewind 5 seconds
        self.set_status("Rewound 5 seconds")

    def fast_forward(self):
        self.playhead = min(self.project_length(), self.playhead + 5.0)  # Fast forward 5 seconds
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
                # Resume playback
                self.play_start_clock = time.perf_counter() - (self.playhead - self.play_start_pos)
                self.paused = False
                self.set_status("Resumed playback")
            else:
                # Pause playback
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
        if self.playing and not self.paused:
            self.stop()
            return
            
        if self.paused:
            # Resume from pause
            self.paused = False
            pygame.mixer.unpause()
            self.play_start_clock = time.perf_counter() - (self.playhead - self.play_start_pos)
            self.set_status("Resumed playback")
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

    def saveas(self):
        """Export current mix to WAV."""
        path = choose_save_file("Save Mix As", "mix.wav")
        if not path:
            self.set_status("Save cancelled.", ok=True)
            return
        mix, start, _ = self.render_mix_with_stems(0.0)
        stereo = to_stereo(mix)
        try:
            import wave
            with wave.open(path, 'wb') as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(SR)
                wf.writeframes(to_int16_stereo(stereo).tobytes())
            self.set_status(f"Saved mix → {path}")
        except Exception as e:
            self.set_status(f"Save failed: {e}", ok=False)

    def save_stems(self):
        folder = choose_folder("Choose folder for stems")
        if not folder:
            self.set_status("Stem export cancelled.", ok=True)
            return
        _, _, stems = self.render_mix_with_stems(0.0)
        try:
            import wave
            for i, (tr, stem) in enumerate(zip(self.tracks, stems)):
                st = to_stereo(stem)
                name = f"{i+1:02d}_{tr.name.replace(' ', '_')}.wav"
                out = os.path.join(folder, name)
                with wave.open(out, 'wb') as wf:
                    wf.setnchannels(2)
                    wf.setsampwidth(2)
                    wf.setframerate(SR)
                    wf.writeframes(to_int16_stereo(st).tobytes())
            self.set_status(f"Exported stems → {folder}")
        except Exception as e:
            self.set_status(f"Stem export failed: {e}", ok=False)

    def update_meters(self):
        if not self.playing or self._stems is None:
            for tr in self.tracks:
                tr.meter_level = 0.0
            return
            
        i = int(self.playhead * SR)
        win = int(0.05 * SR)  # 50ms
        for tr, stem in zip(self.tracks, self._stems):
            if i >= len(stem):
                tr.meter_level = 0.0
                continue
            s = stem[i:i+win]
            tr.meter_level = float(min(1.0, max(0.0, np.sqrt(np.mean((s * tr.volume) ** 2)) * 2.0)))

    def draw(self):
        """Draw the entire UI"""
        self.screen.fill(COL_BG)
        self.draw_topbar()
        self.draw_left_panel()
        self.draw_ruler_and_grid()
        self.draw_tracks()
        self.draw_playhead()
        self.draw_scrollbars()
        
        # Show project name in status bar
        proj_name = os.path.basename(self.project_path) if self.project_path else "Untitled"
        modified = "*" if self.project_modified else ""
        proj_text = f"{proj_name}{modified}"
        txt = FONT_SM.render(proj_text, True, COL_TXT)
        self.screen.blit(txt, (self.WIDTH - 150, self.HEIGHT - self.bottom_bar_h + 4))
        
        txt = FONT_SM.render(self.status, True, self.status_col)
        self.screen.blit(txt, (10, self.HEIGHT - self.bottom_bar_h + 4))

    def draw_topbar(self):
        """Draw the top toolbar with all buttons"""
        # Draw background for button area
        pygame.draw.rect(self.screen, COL_PANEL, (0, 0, self.WIDTH, self.top_bar_h))
        
        # Draw separator line
        pygame.draw.line(self.screen, (60, 60, 80), (0, self.top_bar_h), (self.WIDTH, self.top_bar_h), 2)
        
        mouse = pygame.mouse.get_pos()
        buttons = [self.btn_play, self.btn_pause, self.btn_stop, self.btn_loop,
                  self.btn_rewind, self.btn_ff, self.btn_home, self.btn_end,
                  self.btn_tone, self.btn_save, self.btn_save_stems,
                  self.btn_add_track, self.btn_del_track, self.btn_import, self.btn_delete,
                  self.btn_split, self.btn_zoom_in, self.btn_zoom_out, self.btn_snap,
                  self.btn_new, self.btn_open, self.btn_save_proj, self.btn_save_proj_as]
        
        for b in buttons:
            b.update(mouse)
            b.draw(self.screen)

    def draw_left_panel(self):
        """Draw the left control panel"""
        # Draw left panel background
        panel_rect = pygame.Rect(0, self.top_bar_h, self.left_panel_w, self.HEIGHT - self.top_bar_h)
        pygame.draw.rect(self.screen, COL_PANEL, panel_rect)
        
        # Draw separator line
        pygame.draw.line(self.screen, (60, 60, 80), (self.left_panel_w, self.top_bar_h), 
                         (self.left_panel_w, self.HEIGHT), 2)
        
        # Draw section headers
        section_title = FONT_MD.render("Project Settings", True, COL_TXT)
        self.screen.blit(section_title, (20, self.top_bar_h + 5))
        
        # Draw BPM and time signature labels
        bpm_label = FONT_SM.render("BPM:", True, COL_TXT)
        self.screen.blit(bpm_label, (20, self.top_bar_h + 20))
        
        time_sig_label = FONT_SM.render("Time Signature:", True, COL_TXT)
        self.screen.blit(time_sig_label, (20, self.top_bar_h + 55))
        
        # Draw time signature divider
        divider = FONT_SM.render("/", True, COL_TXT)
        self.screen.blit(divider, (170, self.top_bar_h + 55))
        
        # Draw BPM input fields
        self.bpm_input.draw(self.screen)
        self.time_sig_numerator.draw(self.screen)
        self.time_sig_denominator.draw(self.screen)
        
        # Draw clip controls section header
        clip_title = FONT_MD.render("Clip Controls", True, COL_TXT)
        self.screen.blit(clip_title, (20, self.top_bar_h + 90))
        
        # Draw sliders
        self.sld_gain.draw(self.screen)
        self.sld_speed.draw(self.screen)
        
        # Draw clip info if a clip is selected
        if self.selected_clip:
            # Section header
            info_title = FONT_MD.render("Selected Clip", True, COL_TXT)
            self.screen.blit(info_title, (20, self.top_bar_h + 150))
            
            # Clip info background
            info_rect = pygame.Rect(20, self.top_bar_h + 175, self.left_panel_w - 40, 120)
            pygame.draw.rect(self.screen, COL_BPM_BG, info_rect, border_radius=6)
            pygame.draw.rect(self.screen, (70, 70, 90), info_rect, 2, border_radius=6)
            
            # Clip details
            clip_name_text = FONT_SM.render(f"Name: {self.selected_clip.name}", True, COL_TXT)
            self.screen.blit(clip_name_text, (info_rect.x + 10, info_rect.y + 10))
            
            clip_pos_text = FONT_SM.render(f"Position: {self.selected_clip.start:.2f}s", True, COL_TXT)
            self.screen.blit(clip_pos_text, (info_rect.x + 10, info_rect.y + 30))
            
            clip_len_text = FONT_SM.render(f"Length: {self.selected_clip.eff_len_sec:.2f}s", True, COL_TXT)
            self.screen.blit(clip_len_text, (info_rect.x + 10, info_rect.y + 50))
            
            clip_gain_text = FONT_SM.render(f"Gain: {self.selected_clip.gain:.2f}", True, COL_TXT)
            self.screen.blit(clip_gain_text, (info_rect.x + 10, info_rect.y + 70))
            
            clip_speed_text = FONT_SM.render(f"Speed: {self.selected_clip.speed:.2f}", True, COL_TXT)
            self.screen.blit(clip_speed_text, (info_rect.x + 10, info_rect.y + 90))

    def draw_ruler_and_grid(self):
        """Draw the timeline ruler and grid"""
        y0 = self.top_bar_h
        pygame.draw.rect(self.screen, (28, 28, 38), (self.left_panel_w, y0, self.WIDTH - self.left_panel_w, self.ruler_h))
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
                    pygame.draw.line(self.screen, COL_GRID, (xb, y0 + self.ruler_h//2), (xb, self.HEIGHT - self.bottom_bar_h), 1)
            if x >= self.timeline_origin_x - 30:
                label = FONT_SM.render(str(int(t / bar)), True, COL_TXT)
                self.screen.blit(label, (x - 4, y0 + 4))
            t += bar
        pygame.draw.rect(self.screen, COL_PANEL, (self.left_panel_w, self.top_bar_h + self.ruler_h, self.timeline_origin_x - self.left_panel_w, self.HEIGHT - self.top_bar_h - self.ruler_h))

    def draw_tracks(self):
        """Draw all tracks and their clips"""
        for i, tr in enumerate(self.tracks):
            y = self.track_y(i)
            if y + self.track_h < self.top_bar_h + self.ruler_h or y > self.HEIGHT - self.bottom_bar_h:
                continue
                
            # header
            hdr = pygame.Rect(self.left_panel_w, y, self.timeline_origin_x - self.left_panel_w, self.track_h - 4)
            pygame.draw.rect(self.screen, (30, 30, 40), hdr)
            pygame.draw.rect(self.screen, (55, 55, 70), hdr, 1)
            title_col = COL_TXT if i != self.selected_track else COL_ACC
            self.screen.blit(FONT_MD.render(tr.name, True, title_col), (self.left_panel_w + 10, y + 6))
            
            # Mute button
            tr._mute_rect = pygame.Rect(self.left_panel_w + 12, y + 30, 24, 18)
            pygame.draw.rect(self.screen, COL_MUTE if tr.mute else (70, 70, 80), tr._mute_rect, border_radius=4)
            self.screen.blit(FONT_SM.render("M", True, COL_TXT), tr._mute_rect.move(7, 1))
            
            # Solo button
            tr._solo_rect = pygame.Rect(self.left_panel_w + 42, y + 30, 24, 18)
            pygame.draw.rect(self.screen, COL_SOLO if tr.solo else (70, 70, 80), tr._solo_rect, border_radius=4)
            self.screen.blit(FONT_SM.render("S", True, COL_TXT), tr._solo_rect.move(7, 1))
            
            # Delete track button
            tr._delete_rect = pygame.Rect(self.left_panel_w + 72, y + 30, 24, 18)
            pygame.draw.rect(self.screen, COL_DELETE, tr._delete_rect, border_radius=4)
            self.screen.blit(FONT_SM.render("X", True, COL_TXT), tr._delete_rect.move(7, 1))
            
            # Volume fader
            fx, fw = self.left_panel_w + 100, self.timeline_origin_x - self.left_panel_w - 120
            tr._fader_rect = pygame.Rect(fx, y + 30, fw, 6)
            pygame.draw.rect(self.screen, (60, 60, 80), tr._fader_rect, border_radius=3)
            knob_x = int(fx + tr.volume * (fw - 1))
            pygame.draw.circle(self.screen, COL_ACC2, (knob_x, y + 33), 7)
            
            # meter
            meter = pygame.Rect(self.timeline_origin_x - 20, y + 6, 8, self.track_h - 20)
            pygame.draw.rect(self.screen, COL_METER_BG, meter)
            m_h = int((self.track_h - 20) * max(0.0, min(1.0, tr.meter_level)))
            if m_h > 0:
                pygame.draw.rect(self.screen, COL_METER, (meter.x, meter.bottom - m_h, meter.w, m_h))
                
            # lane
            lane = pygame.Rect(self.timeline_origin_x, y, self.WIDTH - self.timeline_origin_x - self.right_bar_w, self.track_h - 4)
            col = (24, 24, 30) if i % 2 == 0 else (22, 22, 28)
            pygame.draw.rect(self.screen, col, lane)
            
            # clips
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
                # fade handles
                fh = 10
                pts_l = [(rect.x, rect.y), (rect.x + fh, rect.y), (rect.x, rect.y + fh)]
                pts_r = [(rect.right, rect.y), (rect.right - fh, rect.y), (rect.right, rect.y + fh)]
                pygame.draw.polygon(self.screen, COL_FADE, pts_l)
                pygame.draw.polygon(self.screen, COL_FADE, pts_r)

    def draw_playhead(self):
        """Draw the playhead"""
        x = self.time_to_x(self.playhead)
        pygame.draw.line(self.screen, (240, 80, 80), (x, self.top_bar_h), (x, self.HEIGHT - self.bottom_bar_h), 2)

    def draw_scrollbars(self):
        """Draw horizontal and vertical scrollbars"""
        # Horizontal
        y = self.HEIGHT - self.bottom_bar_h - 10
        pygame.draw.rect(self.screen, COL_SCROLL, (self.timeline_origin_x, y, self.WIDTH - self.timeline_origin_x - self.right_bar_w, 10))
        total = max(self.project_length(), self.visible_secs())
        vis = self.visible_secs()
        if total > 0:
            frac = vis / max(total, 1e-9)
            frac = max(0.05, min(1.0, frac))
            # FIX: use correct attribute name timeline_origin_x
            span_px = (self.WIDTH - self.timeline_origin_x - self.right_bar_w) * frac
            max_off = max(0.0, total - vis)
            off_frac = 0.0 if max_off <= 0 else (self.h_off_sec / max_off) * (1 - frac)
            knob_x = int(self.timeline_origin_x + off_frac * (self.WIDTH - self.timeline_origin_x - self.right_bar_w))
            self.h_scroll_rect = pygame.Rect(knob_x, y, int(span_px), 10)
            pygame.draw.rect(self.screen, COL_SCROLL_KNOB, self.h_scroll_rect)
        else:
            self.h_scroll_rect = pygame.Rect(self.timeline_origin_x, y, self.WIDTH - self.timeline_origin_x - self.right_bar_w, 10)

        # Vertical
        x = self.WIDTH - self.right_bar_w
        pygame.draw.rect(self.screen, COL_SCROLL, (x, self.top_bar_h + self.ruler_h, self.right_bar_w, self.HEIGHT - self.top_bar_h - self.ruler_h - self.bottom_bar_h))
        total_px = self.total_track_pixels()
        vis_px = self.HEIGHT - self.top_bar_h - self.ruler_h - self.bottom_bar_h
        if total_px > 0:
            frac = vis_px / max(total_px, 1)
            frac = max(0.05, min(1.0, frac))
            span = int((self.HEIGHT - self.top_bar_h - self.ruler_h - self.bottom_bar_h) * frac)
            max_off = max(0, total_px - vis_px)
            off_frac = 0.0 if max_off <= 0 else (self.v_off_px / max_off) * (1 - frac)
            knob_y = int(self.top_bar_h + self.ruler_h + off_frac * (self.HEIGHT - self.top_bar_h - self.ruler_h - self.bottom_bar_h))
            self.v_scroll_rect = pygame.Rect(x, knob_y, self.right_bar_w, span)
            pygame.draw.rect(self.screen, COL_SCROLL_KNOB, self.v_scroll_rect)
        else:
            self.v_scroll_rect = pygame.Rect(x, self.top_bar_h + self.ruler_h, self.right_bar_w, self.HEIGHT - self.top_bar_h - self.ruler_h - self.bottom_bar_h)

    def wheel(self, e):
        """Handle mouse wheel events"""
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
        """Handle keyboard events"""
        mods = pygame.key.get_mods()
        ctrl_or_cmd = (mods & pygame.KMOD_CTRL) or (mods & pygame.KMOD_META)

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

        elif e.key == pygame.K_LEFT:
            self.playhead = max(0.0, self.playhead - 1.0)

        elif e.key == pygame.K_RIGHT:
            self.playhead = min(self.project_length(), self.playhead + 1.0)

        elif e.key == pygame.K_z:
            if self.selected_clip:
                l, r = self.selected_clip.bounds()
                mid = (l + r) / 2
                span = max(0.25, r - l)
                self.set_zoom(max(60, min(400, (self.WIDTH - self.timeline_origin_x) / span)), anchor_time=mid)
                self.h_off_sec = max(0.0, l - 0.25 * span)

        # Fixed conflicting shortcuts:
        elif ctrl_or_cmd and (mods & pygame.KMOD_SHIFT) and e.key == pygame.K_s:
            # Ctrl+Shift+S → Save Project As
            self.save_project_as()

        elif ctrl_or_cmd and e.key == pygame.K_s:
            # Ctrl+S → Save Project
            self.save_project()

        elif ctrl_or_cmd and e.key == pygame.K_m:
            # Ctrl+M → Export Mix
            self.saveas()

        elif ctrl_or_cmd and e.key == pygame.K_e:
            # Ctrl+E → Export Stems
            self.save_stems()

        elif ctrl_or_cmd and e.key == pygame.K_l:
            self.toggle_loop()

        elif ctrl_or_cmd and e.key == pygame.K_n:
            self.new_project()

        elif ctrl_or_cmd and e.key == pygame.K_o:
            self.open_project()

    def mouse_down(self, e):
        """Handle mouse button down events"""
        mx, my = e.pos
        self.drag_last_mouse = (mx, my)

        # Check if any button was clicked
        mouse_buttons = [self.btn_play, self.btn_pause, self.btn_stop, self.btn_loop,
                         self.btn_rewind, self.btn_ff, self.btn_home, self.btn_end,
                         self.btn_tone, self.btn_save, self.btn_save_stems,
                         self.btn_add_track, self.btn_del_track, self.btn_import, self.btn_delete,
                         self.btn_split, self.btn_zoom_in, self.btn_zoom_out, self.btn_snap,
                         self.btn_new, self.btn_open, self.btn_save_proj, self.btn_save_proj_as]
        
        for button in mouse_buttons:
            if button.rect.collidepoint(mx, my):
                button.handle_click()
                return

        # Check if text inputs were clicked
        if self.bpm_input.rect.collidepoint(mx, my):
            self.bpm_input.active = True
            return
        if self.time_sig_numerator.rect.collidepoint(mx, my):
            self.time_sig_numerator.active = True
            return
        if self.time_sig_denominator.rect.collidepoint(mx, my):
            self.time_sig_denominator.active = True
            return

        # click track header selects track
        top_tracks = self.top_bar_h + self.ruler_h
        if my >= top_tracks and self.left_panel_w <= mx < self.timeline_origin_x:
            ti = self._ensure_track_for_y(my)
            self.selected_track = ti
            # fader/mute/solo/delete hit-testing
            tr = self.tracks[ti]
            if getattr(tr, "_fader_rect", None) and tr._fader_rect.collidepoint(mx, my):
                self.drag_mode = 'fader'; self.drag_fader_track = ti; return
            if getattr(tr, "_mute_rect", None) and tr._mute_rect.collidepoint(mx, my):
                tr.mute = not tr.mute; self.invalidate_render(); self.mark_project_modified(); return
            if getattr(tr, "_solo_rect", None) and tr._solo_rect.collidepoint(mx, my):
                tr.solo = not tr.solo; self.invalidate_render(); self.mark_project_modified(); return
            if getattr(tr, "_delete_rect", None) and tr._delete_rect.collidepoint(mx, my):
                self.delete_track(); return
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
        """Handle fader drag events"""
        tr = self.tracks[self.drag_fader_track]
        rel = (mx - tr._fader_rect.x) / max(1, tr._fader_rect.w - 1)
        tr.volume = max(0.0, min(1.5, rel))
        self.invalidate_render()
        self.mark_project_modified()

    def mouse_up(self, e):
        """Handle mouse button up events"""
        if self.drag_mode in ('move', 'trim_l', 'trim_r', 'fade_l', 'fade_r'):
            self.mark_project_modified()
            
        self.drag_mode = None
        self.drag_clip_ref = None
        self.drag_fader_track = None
        self._drag_scroll_h = False
        self._drag_scroll_v = False

    def mouse_motion(self, e):
        """Handle mouse motion events"""
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
            track_w = self.WIDTH - self.timeline_origin_x - self.right_bar_w
            frac = vis / total if total else 1.0
            denom = max(1, int((1 - frac) * track_w))
            self.h_off_sec = max(0.0, min(max_off, self.h_off_sec + (dx / denom) * max_off))
            return

        if self._drag_scroll_v:
            total_px = self.total_track_pixels()
            vis_px = self.HEIGHT - self.top_bar_h - self.ruler_h - self.bottom_bar_h
            max_off = max(0, total_px - vis_px)
            track_h = self.HEIGHT - self.top_bar_h - self.ruler_h - self.bottom_bar_h
            frac = vis_px / total_px if total_px else 1.0
            denom = max(1, int((1 - frac) * track_h))
            self.v_off_px = int(max(0, min(max_off, self.v_off_px + (dy / denom) * max_off)))
            return

        if self.drag_mode == 'pan':
            self.h_off_sec = max(0.0, self.h_off_sec - dx / self.px_per_sec)
            total_px = self.total_track_pixels()
            vis_px = self.HEIGHT - self.top_bar_h - self.ruler_h - self.bottom_bar_h
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

    def update(self, dt_ms, events):
        """Update the editor state"""
        # Update text inputs
        self.bpm_input.update(events, dt_ms)
        self.time_sig_numerator.update(events, dt_ms)
        self.time_sig_denominator.update(events, dt_ms)
        
        # Update BPM from input field
        try:
            new_bpm = float(self.bpm_input.text)
            if new_bpm != self.bpm and 60 <= new_bpm <= 200:
                self.bpm = new_bpm
                self.mark_project_modified()
        except ValueError:
            pass
            
        if self.selected_clip and self.drag_mode is None:
            g = float(self.sld_gain.val)
            s = float(self.sld_speed.val)
            if abs(self.selected_clip.gain - g) > 1e-6 or abs(self.selected_clip.speed - s) > 1e-6:
                self.selected_clip.gain = g
                self.selected_clip.speed = s
                self.invalidate_render()
                self.mark_project_modified()
                
        if self.playing and self.play_start_clock is not None and not self.paused:
            elapsed = time.perf_counter() - self.play_start_clock
            self.playhead = self.play_start_pos + elapsed
            if self.play_end_sec is not None and self.playhead >= self.play_end_sec:
                if self.looping:
                    self.playhead = 0.0
                    self.play_start_pos = 0.0
                    self.play_start_clock = time.perf_counter()
                    self.play()
                else:
                    self.stop()
            if self.playhead > self.h_off_sec + 0.85 * self.visible_secs():
                self.h_off_sec = max(0.0, self.playhead - 0.85 * self.visible_secs())
        self.update_meters()

    def handle_event(self, e):
        """Handle pygame events"""
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

    def new_project(self):
        """Create a new empty project."""
        if self.project_modified:
            # TODO: Add confirmation dialog
            pass
            
        self.tracks = []
        self._create_track()
        self._create_track()
        self.selected_track = 0
        self.select_clip(None)
        self.playhead = 0.0
        self.project_path = None
        self.project_modified = False
        self.audio_files = {}
        self.invalidate_render()
        self.set_status("New project created")

    def open_project(self):
        """Open a project file."""
        path = choose_open_file("Open Project", filetypes=[("PyDAW Project", "*.pdaw"), ("All files", "*.*")])
        if not path:
            return
            
        try:
            with open(path, 'rb') as f:
                project_data = pickle.load(f)
                
            # Clear current project
            self.tracks = []
            self.audio_files = project_data.get('audio_files', {})
            
            # Recreate tracks
            for track_data in project_data.get('tracks', []):
                track = Track.from_dict(track_data)
                self.tracks.append(track)
                
                # Recreate clips
                for clip_data in track_data.get('clips', []):
                    audio_key = clip_data.get('audio_key', '')
                    audio = self.audio_files.get(audio_key, np.array([], dtype=np.float32))
                    clip = Clip.from_dict(clip_data, audio)
                    track.clips.append(clip)
            
            # Restore project settings
            self.bpm = project_data.get('bpm', 120.0)
            self.bpm_input.text = str(self.bpm)
            self.snap = project_data.get('snap', True)
            self.btn_snap.state = self.snap
            self.looping = project_data.get('looping', False)
            self.btn_loop.state = self.looping
            
            self.project_path = path
            self.project_modified = False
            self.select_clip(None)
            self.invalidate_render()
            self.set_status(f"Project loaded: {os.path.basename(path)}")
            
        except Exception as e:
            self.set_status(f"Failed to load project: {e}", ok=False)

    def save_project(self):
        """Save the current project."""
        if self.project_path:
            self._save_project_to_path(self.project_path)
        else:
            self.save_project_as()

    def save_project_as(self):
        """Save the current project with a new name."""
        path = choose_save_file("Save Project As", default_name="project.pdaw", 
                               filetypes=[("PyDAW Project", "*.pdaw")])
        if not path:
            return
            
        self._save_project_to_path(path)

    def _save_project_to_path(self, path: str):
        """Save project to the specified path."""
        try:
            # Prepare project data
            project_data = {
                'tracks': [track.to_dict() for track in self.tracks],
                'bpm': self.bpm,
                'snap': self.snap,
                'looping': self.looping,
                'audio_files': self.audio_files
            }
            
            with open(path, 'wb') as f:
                pickle.dump(project_data, f)
                
            self.project_path = path
            self.project_modified = False
            self.set_status(f"Project saved: {os.path.basename(path)}")
            
        except Exception as e:
            self.set_status(f"Failed to save project: {e}", ok=False)

    def mark_project_modified(self):
        """Mark the project as modified."""
        self.project_modified = True
