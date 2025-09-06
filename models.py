from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pygame

SR = 44100


class Clip:
    def __init__(
        self,
        audio: np.ndarray,
        name: str,
        start: float = 0.0,
        speed: float = 1.0,
        gain: float = 1.0,
        in_pos: float = 0.0,
        out_pos: Optional[float] = None,
        fade_in: float = 0.02,
        fade_out: float = 0.02,
    ):
        self.audio = audio.astype(np.float32)
        self.name = name
        self.start = float(start)     # timeline start (sec)
        self.speed = float(speed)     # 1.0 normal; 0.5 slower; 1.5 faster
        self.gain = float(gain)
        self.in_pos = float(in_pos)   # seconds into source
        self.out_pos = float(out_pos) if out_pos is not None else (len(audio) / SR)
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
        right_clip = Clip(
            self.audio,
            name=self.name + " (split)",
            start=t_sec,
            speed=self.speed,
            gain=self.gain,
            in_pos=split_src,
            out_pos=self.out_pos,
            fade_in=self.fade_in,
            fade_out=self.fade_out,
        )
        right_clip.track_index = self.track_index
        self.out_pos = split_src
        return right_clip

    # ----- serialization helpers -----
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "start": self.start,
            "speed": self.speed,
            "gain": self.gain,
            "in_pos": self.in_pos,
            "out_pos": self.out_pos,
            "fade_in": self.fade_in,
            "fade_out": self.fade_out,
            "track_index": self.track_index,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], audio: Optional[np.ndarray] = None) -> "Clip":
        """Create a clip from dictionary data."""
        if audio is None:
            audio = np.array([], dtype=np.float32)
        clip = cls(
            audio=audio,
            name=data["name"],
            start=float(data["start"]),
            speed=float(data["speed"]),
            gain=float(data["gain"]),
            in_pos=float(data["in_pos"]),
            out_pos=float(data["out_pos"]),   # FIXED: was 'out_out'
            fade_in=float(data["fade_in"]),
            fade_out=float(data["fade_out"]),
        )
        clip.track_index = int(data.get("track_index", 0))
        return clip


class Track:
    def __init__(self, name: str):
        self.name = name
        self.volume = 1.0
        self.mute = False
        self.solo = False
        self.clips: List[Clip] = []
        # runtime/UI
        self.meter_level = 0.0       # 0..1 (post-fader)
        self._fader_rect: Optional[pygame.Rect] = None
        self._mute_rect: Optional[pygame.Rect] = None
        self._solo_rect: Optional[pygame.Rect] = None
        self._delete_rect: Optional[pygame.Rect] = None

    # ----- serialization helpers -----
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "volume": self.volume,
            "mute": self.mute,
            "solo": self.solo,
            "clips": [c.to_dict() for c in self.clips],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Track":
        track = cls(data["name"])
        track.volume = float(data["volume"])
        track.mute = bool(data["mute"])
        track.solo = bool(data["solo"])
        return track
