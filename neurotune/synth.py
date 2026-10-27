"""Sound side: renders audio block by block and drops each phase-locked
pulse in at the exact sample it was scheduled for.

Two modes:
  sleep - Elemind-style: 1/f pink-noise bursts (about a quarter of an alpha
          cycle long) over quiet rain, 18 dB below the pulses.
  music - the same phase-locked onsets, but each pulse is a soft glassy
          bell on a chord tone, over a wide, slowly drifting sine pad whose
          brightness follows your relaxation and which breathes at 6
          breaths/min. Drowsiness darkens and quietens it. Pad and bells
          share a light stereo reverb.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .brain import Brain, Pulse

# Paul Kellet-style 1/f filter for pink noise.
_PINK_B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
_PINK_A = [1, -2.494956002, 2.017265875, -0.522189400]

def _hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)

# D Dorian, slow progression: Dm9, Bbmaj9(#11), Fmaj9/A, C6/9/G.
# Open voicings (no semitone rubs) with E4 held on top of every chord, so
# changes glide instead of jumping. MIDI notes; the UI's pad octave lifts them.
CHORDS = [
    [38, 45, 53, 60, 64],   # D2  A2 F3 C4 E4
    [34, 41, 50, 57, 64],   # Bb1 F2 D3 A3 E4
    [33, 41, 48, 55, 64],   # A1  F2 C3 G3 E4
    [43, 50, 57, 60, 64],   # G2  D3 A3 C4 E4
]
CHORD_SECONDS = 20.0
XFADE = 0.4                 # last 40% of each chord cross-fades into the next
# Bell walk over the upper chord tones: up and back down, never a leap.
BELL_STEPS = [1, 2, 3, 4, 3, 2]


class _Reverb:
    """Small stereo Schroeder/Freeverb-style reverb: 4 combs + 2 allpasses per
    channel. Vectorised in chunks no longer than the shortest delay."""
    COMBS = (1557, 1617, 1491, 1422)
    ALLPASS = (556, 441)
    SPREAD = 23

    def __init__(self, sr, feedback=0.86):
        k = sr / 44100
        self.g = feedback
        self.combs = [[int(d * k) + c * self.SPREAD for d in self.COMBS] for c in (0, 1)]
        self.aps = [[int(d * k) + c * self.SPREAD for d in self.ALLPASS] for c in (0, 1)]
        self.comb_hist = [[np.zeros(d) for d in ch] for ch in self.combs]
        self.ap_hist = [[np.zeros(d) for d in ch] for ch in self.aps]
        self.chunk = min(min(ch) for ch in self.aps)

    @staticmethod
    def _step(hist, x, g, allpass):
        m = len(x)
        d = hist[:m]
        v = x + g * d
        hist = np.concatenate([hist[m:], v])
        return (d - x if allpass else v), hist

    def process(self, x):
        """x: (n, 2) dry input -> (n, 2) wet output."""
        out = np.zeros_like(x)
        for c in (0, 1):
            for s in range(0, len(x), self.chunk):
                xin = x[s:s + self.chunk, c]
                acc = np.zeros(len(xin))
                for i in range(len(self.combs[c])):
                    y, self.comb_hist[c][i] = self._step(self.comb_hist[c][i], xin, self.g, False)
                    acc += y
                acc *= 0.25
                for i in range(len(self.aps[c])):
                    acc, self.ap_hist[c][i] = self._step(self.ap_hist[c][i], acc, 0.5, True)
                out[s:s + self.chunk, c] = acc
        return out


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
        # Phase per chord slot, voice and ear, so a chord keeps its own phase
        # through the cross-fade and never jumps when it takes over.
        self._pad_phase = np.zeros((len(CHORDS), len(CHORDS[0]), 2))
        self._pad_lp = np.zeros((2, 2))     # two one-pole stages x two ears
        self._cutoff = 800.0                # smoothed, so brain-state changes don't zipper
        self._breath_phase = 0.0
        self._reverb = _Reverb(sr)
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
            shimmer_decay=0.35,   # bell ring time constant, s
            rain_db=0.0,          # background noise level
            reverb=0.35,          # wet level on pad + bells (music mode)
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
            return np.column_stack([burst, burst])
        # music: a soft glassy bell on the current chord, high register. The
        # 5 ms raised-cosine attack still gives a clear onset at the scheduled sample.
        chord = self._current_chord()
        step = BELL_STEPS[self._shimmer_i % len(BELL_STEPS)]
        note = chord[step] + 24 + 12 * self.mix["shimmer_octave"]
        pan = 0.5 + 0.25 * np.sin(self._shimmer_i * 2.4)      # drift gently across the stereo field
        self._shimmer_i += 1
        f0 = _hz(note)
        decay = self.mix["shimmer_decay"]
        n = int((0.005 + 5 * decay) * self.sr)
        t = np.arange(n) / self.sr
        env = np.exp(-t / decay)
        k = int(0.005 * self.sr)
        env[:k] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, k))
        tone = (np.sin(2 * np.pi * f0 * 0.9983 * t) + np.sin(2 * np.pi * f0 * 1.0017 * t)   # +-3 cent chorus
                + 0.3 * np.sin(2 * np.pi * 2 * f0 * t) * np.exp(-t / (0.4 * decay))
                + 0.08 * np.sin(2 * np.pi * 3 * f0 * t) * np.exp(-t / (0.2 * decay)))
        y = tone * env * 0.11 * 10 ** (self.mix["shimmer_db"] / 20)
        return np.column_stack([y * np.sqrt(1 - pan), y * np.sqrt(pan)])

    def _current_chord(self):
        return CHORDS[int(self.t_audio // self.mix["chord_seconds"]) % len(CHORDS)]

    def _pad(self, n):
        """Wide sine pad: each ear detuned the other way (slow chorus between
        the ears), each voice drifting in level, equal-power chord cross-fades,
        and a gentle two-pole low-pass. Returns (n, 2)."""
        st = self.brain.state
        t = self.t_audio
        cs = self.mix["chord_seconds"]
        i = int(t // cs)
        frac = (t % cs) / cs
        xf = np.clip((frac - (1 - XFADE)) / XFADE, 0, 1)
        tt = t + np.arange(n) / self.sr
        out = np.zeros((n, 2))
        for ci, w in ((i, np.cos(xf * np.pi / 2)), (i + 1, np.sin(xf * np.pi / 2))):
            if w <= 1e-4:
                continue
            slot = ci % len(CHORDS)
            chord = CHORDS[slot]
            for v, m in enumerate(chord):
                drift = 0.8 + 0.2 * np.sin(2 * np.pi * (0.05 + 0.017 * v) * tt + 1.3 * v)
                gain = w * (0.75 if v == 0 else 1.0) / len(chord)   # keep the bass soft
                for ear, det in enumerate((-0.05, 0.05)):
                    f = _hz(m + det + 12 * self.mix["pad_octave"])
                    ph = self._pad_phase[slot, v, ear] + 2 * np.pi * f * np.arange(1, n + 1) / self.sr
                    self._pad_phase[slot, v, ear] = ph[-1] % (2 * np.pi)
                    out[:, ear] += gain * drift * (np.sin(ph) + 0.06 * np.sin(2 * ph))
        # brightness follows relaxation, darkens with drowsiness (smoothed over ~1.5 s)
        target = min((400 + 2200 * st.relax * (1 - 0.7 * st.drowsy)) * self.mix["brightness"], 0.45 * self.sr)
        self._cutoff += (target - self._cutoff) * min(1.0, n / (1.5 * self.sr))
        a = np.exp(-2 * np.pi * self._cutoff / self.sr)
        for stage in range(2):
            for ear in range(2):
                out[:, ear], zf = signal.lfilter([1 - a], [1, -a], out[:, ear], zi=[self._pad_lp[stage, ear]])
                self._pad_lp[stage, ear] = zf[0]
        bph = self._breath_phase + 2 * np.pi * self.mix["breath_bpm"] / 60 * np.arange(n) / self.sr
        self._breath_phase = (bph[-1] + 2 * np.pi * self.mix["breath_bpm"] / 60 / self.sr) % (2 * np.pi)
        breath = 0.72 + 0.28 * np.sin(bph)   # default 6 breaths/min
        return out * breath[:, None] * 0.34 * 10 ** (self.mix["pad_db"] / 20)

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
        rain = rain / self._pink_scale * 0.3 * 10 ** ((self.mix["rain_db"] - 18) / 20)   # rain 18 dB under the pulses
        music = self.mode == "music"
        dry = self._pad(n) if music else np.zeros((n, 2))

        while self._inject:
            tone = self._inject.pop(0)
            self.voices.append([np.column_stack([tone, tone]), 0])

        keep = []
        for v in self.voices:
            buf, pos = v
            start = max(0, -pos)
            src0 = max(0, pos)
            m = min(n - start, len(buf) - src0)
            if m > 0:
                dry[start:start + m] += buf[src0:src0 + m]
            v[1] = pos + n
            if v[1] < len(buf):
                keep.append(v)
        self.voices = keep

        if music:
            # rain sits further back in music mode; pad and bells get the reverb
            mix = dry + self.mix["reverb"] * self._reverb.process(dry) + 0.3 * rain[:, None]
        else:
            mix = dry + rain[:, None]

        # master fade (sleep detected or session ending)
        st = self.brain.state
        target = 0.0 if (self.fading or st.asleep) else 1.0 - 0.5 * st.drowsy * (self.mode == "music")
        step = n / self.sr / 60.0          # 60 s fades
        self.master += np.clip(target - self.master, -step, step)
        self.t_audio += n / self.sr
        out = mix * self.gain * self.master
        # music: soft limiter instead of a hard clip; sleep: unchanged
        out = np.tanh(out) if music else np.clip(out, -1, 1)
        return out.astype(np.float32)
