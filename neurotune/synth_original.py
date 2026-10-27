"""Sound side: renders audio block by block and drops each phase-locked
pulse in at the exact sample it was scheduled for.

Two modes:
  sleep - Elemind-style: 1/f pink-noise bursts (about a quarter of an alpha
          cycle long) over quiet rain, 18 dB below the pulses.
  music - the same phase-locked onsets, but each pulse is a pitched
          "shimmer" (pink noise through a resonant filter on a chord tone),
          over a slow pad whose brightness follows your relaxation and
          which breathes at 6 breaths/min. Drowsiness darkens and quietens it.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .brain import Brain, Pulse

# Paul Kellet-style 1/f filter for pink noise.
_PINK_B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
_PINK_A = [1, -2.494956002, 2.017265875, -0.522189400]

# D Dorian, slow progression: Dm9, Bbmaj7, Fmaj7/A, C6/G (Hz of voicing)
def _hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)

CHORDS = [
    [50, 57, 60, 64, 65],   # D2 A2 C3 E3 F3
    [46, 53, 57, 62, 65],   # Bb1 F2 A2 D3 F3
    [45, 53, 57, 60, 64],   # A1 F2 A2 C3 E3
    [43, 52, 55, 60, 62],   # G1 E2 G2 C3 D3
]
CHORD_SECONDS = 16.0


def _ramp(n, sr, ms=2.0):
    """Raised-cosine on/off ramps (avoids clicks that would blur the onset)."""
    k = max(1, int(sr * ms / 1000))
    env = np.ones(n)
    r = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, k))
    env[:k] = r
    env[-k:] = r[::-1]
    return env


class Synth:
    def __init__(self, brain: Brain, sr: int = 48000, mode: str = "sleep",
                 volume_db: float = -12.0, seed: int = 0):
        self.brain = brain
        self.sr = sr
        self.mode = mode
        self.gain = 10 ** (volume_db / 20)
        self.rng = np.random.default_rng(seed)
        self._pink_zi = signal.lfilter_zi(_PINK_B, _PINK_A) * 0
        self._rain_sos = signal.butter(2, 2500, "lowpass", fs=sr, output="sos")
        self._rain_zi = np.zeros((self._rain_sos.shape[0], 2))
        bank = self._pink(sr * 4)
        self._pink_scale = float(np.std(bank))
        self._pink_bank = bank / self._pink_scale * 0.3   # pulse level: 0.3 rms
        self.voices: list[list] = []   # [buffer, position]
        self.t_audio = 0.0             # seconds of audio rendered (for LFOs)
        self.master = 1.0              # fades out when asleep / stopped
        self.fading = False
        self.log: list[dict] = []
        self._shimmer_i = 0
        self._pad_phase = np.zeros((len(CHORDS[0]), 2))
        self._pad_lp = np.zeros(2)
        self._breath_phase = 0.0
        self._inject: list[np.ndarray] = []   # one-shot sounds queued from other threads
        self.started = True
        # Live mix (the UI changes these while audio runs; defaults = original sound).
        self.mix = dict(
            pad_db=0.0,           # pad level
            pad_octave=0,         # pad transposition (+1/+2 helps small speakers)
            brightness=1.0,       # multiplies the relax-driven pad cutoff
            breath_bpm=6.0,       # pad swell rate
            chord_seconds=CHORD_SECONDS,
            shimmer_db=0.0,       # brain-locked pulse level (music mode)
            shimmer_octave=0,
            shimmer_decay=0.08,   # shimmer ring time constant, s
            rain_db=0.0,          # background noise level
        )

    @property
    def volume_db(self) -> float:
        return 20 * np.log10(max(self.gain, 1e-9))

    @volume_db.setter
    def volume_db(self, db: float):
        self.gain = 10 ** (db / 20)

    def beep(self, hz: float = 440.0, secs: float = 0.6):
        """Queue a test tone (thread-safe enough: list.append is atomic)."""
        n = int(secs * self.sr)
        tone = 0.5 * np.sin(2 * np.pi * hz * np.arange(n) / self.sr) * _ramp(n, self.sr, 10)
        self._inject.append(tone)

    # ---------------------------------------------------------------- utils
    def _pink(self, n):
        w = self.rng.standard_normal(n)
        y, self._pink_zi = signal.lfilter(_PINK_B, _PINK_A, w, zi=self._pink_zi)
        return y

    def _pulse_buffer(self, p: Pulse) -> np.ndarray:
        n = max(int(p.duration * self.sr), 16)
        start = self.rng.integers(0, len(self._pink_bank) - n)
        burst = self._pink_bank[start:start + n] * _ramp(n, self.sr)
        if self.mode == "sleep":
            return burst
        # music: pitched shimmer on the current chord, high register
        chord = self._current_chord()
        note = (chord[1 + self._shimmer_i % (len(chord) - 1)] + 24 + 12 * ((self._shimmer_i // 4) % 2)
                + 12 * self.mix["shimmer_octave"])
        self._shimmer_i += 1
        f0 = _hz(note)
        b, a = signal.iirpeak(f0, Q=40, fs=self.sr)
        decay = self.mix["shimmer_decay"]
        tail = int(max(0.25, 4 * decay) * self.sr)
        x = np.concatenate([burst, np.zeros(tail)])
        y = signal.lfilter(b, a, x)
        y *= np.exp(-np.arange(len(y)) / (decay * self.sr))
        y /= (np.abs(y).max() + 1e-9)
        return y * 0.45 * 10 ** (self.mix["shimmer_db"] / 20)

    def _current_chord(self):
        return CHORDS[int(self.t_audio // self.mix["chord_seconds"]) % len(CHORDS)]

    def _pad(self, n):
        """Detuned sine pad with a one-pole low-pass; cross-fades chords."""
        st = self.brain.state
        t = self.t_audio
        cs = self.mix["chord_seconds"]
        i = int(t // cs)
        frac = (t % cs) / cs
        xf = np.clip((frac - 0.75) / 0.25, 0, 1)   # last 4 s of each chord: fade into next
        out = np.zeros(n)
        for ci, w in ((i, 1 - xf), (i + 1, xf)):
            if w <= 0:
                continue
            chord = CHORDS[ci % len(CHORDS)]
            for v, m in enumerate(chord):
                for d, det in enumerate((-0.07, 0.07)):
                    f = _hz(m + det + 12 * self.mix["pad_octave"])
                    ph = self._pad_phase[v, d] + 2 * np.pi * f * np.arange(1, n + 1) / self.sr
                    if ci == i:
                        self._pad_phase[v, d] = ph[-1] % (2 * np.pi)
                    s = np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.1 * np.sin(3 * ph)
                    out += w * s / len(chord)
        # brightness follows relaxation, darkens with drowsiness
        cutoff = min((300 + 2200 * st.relax * (1 - 0.7 * st.drowsy)) * self.mix["brightness"], 0.45 * self.sr)
        a = np.exp(-2 * np.pi * cutoff / self.sr)
        y, zf = signal.lfilter([1 - a], [1, -a], out, zi=[self._pad_lp[0]])
        self._pad_lp[0] = zf[0]
        bph = self._breath_phase + 2 * np.pi * self.mix["breath_bpm"] / 60 * np.arange(n) / self.sr
        self._breath_phase = (bph[-1] + 2 * np.pi * self.mix["breath_bpm"] / 60 / self.sr) % (2 * np.pi)
        breath = 0.65 + 0.35 * np.sin(bph)   # default 6 breaths/min
        return y * breath * 0.3 * 10 ** (self.mix["pad_db"] / 20)

    # --------------------------------------------------------------- render
    def render(self, n: int, block_start: float) -> np.ndarray:
        """Render n stereo frames whose first frame plays at block_start."""
        block_end = block_start + n / self.sr
        while True:
            p = self.brain.take_pulse(block_end)
            if p is None:
                break
            idx = int(round((p.t - block_start) * self.sr))
            late = idx < 0
            idx = max(idx, 0)
            onset = block_start + idx / self.sr
            if late and (block_start - p.t) > 0.003:
                self.log.append({"t": p.t, "played": False, "freq": p.freq, "env": p.env})
                continue
            self.voices.append([self._pulse_buffer(p), -idx])
            self.log.append({"t": onset, "played": True, "freq": p.freq, "env": p.env,
                             "dur": p.duration})

        pink = self._pink(n)
        rain, self._rain_zi = signal.sosfilt(self._rain_sos, pink, zi=self._rain_zi)
        mono = rain / self._pink_scale * 0.3 * 10 ** ((self.mix["rain_db"] - 18) / 20)   # rain 18 dB under the pulses
        if self.mode == "music":
            mono = mono * 0.5 + self._pad(n)

        while self._inject:
            self.voices.append([self._inject.pop(0), 0])

        keep = []
        for v in self.voices:
            buf, pos = v
            start = max(0, -pos)
            src0 = max(0, pos)
            m = min(n - start, len(buf) - src0)
            if m > 0:
                mono[start:start + m] += buf[src0:src0 + m]
            v[1] = pos + n
            if v[1] < len(buf):
                keep.append(v)
        self.voices = keep

        # master fade (sleep detected or session ending)
        st = self.brain.state
        target = 0.0 if (self.fading or st.asleep) else 1.0 - 0.5 * st.drowsy * (self.mode == "music")
        step = n / self.sr / 60.0          # 60 s fades
        self.master += np.clip(target - self.master, -step, step)
        self.t_audio += n / self.sr
        out = np.clip(mono * self.gain * self.master, -1, 1)
        return np.column_stack([out, out]).astype(np.float32)
