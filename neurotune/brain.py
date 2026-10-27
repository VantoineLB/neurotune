"""Brain side: track alpha phase, decide when the next pulse should start,
and follow slower brain-state trends (relaxation, drowsiness, sleep).

Clock-agnostic: every method takes explicit timestamps in seconds on one
shared clock (pylsl.local_clock() in real time, a virtual clock in the
simulator), so the same code runs live and offline.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .dsp import ECHT, TWO_PI, band_powers, wrap


@dataclass
class Settings:
    fs: float = 256.0                 # EEG sample rate (Muse = 256 Hz)
    iaf: float = 10.0                 # individual alpha frequency, Hz
    band_halfwidth: float = 1.5       # tracking band = iaf +/- this
    window_s: float = 1.0             # ecHT window length
    target: str = "trough"            # where the evoked P1 should land: trough | peak | <degrees onset>
    p1_latency_s: float = 0.060       # sound onset -> P1 evoked response
    eeg_latency_s: float = 0.0        # how late EEG timestamps are vs reality (measure with `loopback`)
    phase_bias_deg: float = 0.0       # residual correction from `analyze`
    gate: float = 0.8                 # stimulate only when alpha envelope > gate * running median (0 = always)
    min_gap_cycles: float = 0.75      # never two pulses closer than this
    freq_adapt: float = 0.05          # how fast the tracked frequency follows the data
    pulse_cycles: float = 0.25        # pulse length as a fraction of one alpha cycle (Elemind: 90 deg)


@dataclass
class Pulse:
    t: float          # onset time on the shared clock (sound leaves the driver)
    freq: float       # tracked alpha frequency when scheduled
    env: float        # alpha envelope when scheduled
    duration: float   # seconds


@dataclass
class State:
    relax: float = 0.5        # 0..1, alpha share of alpha+beta, normalised to the session
    drowsy: float = 0.0       # 0..1, theta/alpha rise vs baseline
    alpha_rel: float = 1.0    # alpha power vs first-minutes baseline
    asleep: bool = False
    history: list = field(default_factory=list)


class Brain:
    def __init__(self, s: Settings):
        self.s = s
        self.n = int(round(s.window_s * s.fs))
        self.echt = ECHT(s.fs, self.n, (s.iaf - s.band_halfwidth, s.iaf + s.band_halfwidth))
        self.buf = deque(maxlen=int(s.fs * 8))     # raw samples for ecHT and band power
        self.tbuf = deque(maxlen=int(s.fs * 8))
        self.envs = deque(maxlen=int(s.fs * 10 / 12) + 1)  # one per update, ~10 s
        self.freq = s.iaf
        self._last_phase = None
        self._last_t = None
        self.pending: Pulse | None = None
        self.last_onset = -1e9
        self.lock = threading.Lock()
        self.state = State()
        self._baseline = []
        self._last_state_t = None
        self.enabled = True
        self.updates = 0

    # ------------------------------------------------------------------ phase
    def onset_phase(self, freq: float) -> float:
        """Alpha phase at which a pulse must start so P1 lands on the target."""
        t = self.s.target
        if t in ("trough", "peak"):
            land = np.pi if t == "trough" else 0.0
            return float(wrap(land - TWO_PI * freq * self.s.p1_latency_s))
        return float(np.deg2rad(float(t)))

    def feed(self, samples: np.ndarray, stamps: np.ndarray, now: float, earliest: float):
        """Add new EEG samples (1-D, the chosen channel) with timestamps.

        now      : current time on the shared clock
        earliest : earliest onset the audio side can still honour
        """
        # The Muse sends NaN when a sensor loses contact. Hold the last good value:
        # one NaN in the window would otherwise end up in self.freq and stay there.
        samples = np.asarray(samples, dtype=float)
        bad = ~np.isfinite(samples)
        if bad.any():
            last = self.buf[-1] if self.buf else 0.0
            idx = np.maximum.accumulate(np.where(~bad, np.arange(len(samples)), -1))
            samples = np.where(idx >= 0, samples[np.maximum(idx, 0)], last)
        self.buf.extend(samples)
        self.tbuf.extend(stamps)
        if len(self.buf) < self.n:
            return
        x = np.fromiter(self.buf, float, len(self.buf))[-self.n:]
        t_last = self.tbuf[-1] - self.s.eeg_latency_s
        phase, env = self.echt.last(x)
        phase += np.deg2rad(self.s.phase_bias_deg)
        self.updates += 1
        self.envs.append(env)

        # Follow slow drifts in alpha frequency (clamped near the IAF).
        if self._last_phase is not None:
            dt = t_last - self._last_t
            if 0 < dt < 0.2:
                expected = self.freq * dt * TWO_PI
                dphi = (phase - self._last_phase - expected + np.pi) % TWO_PI - np.pi + expected
                f_inst = dphi / (TWO_PI * dt)
                if self.s.iaf - 2 < f_inst < self.s.iaf + 2:
                    self.freq += self.s.freq_adapt * (f_inst - self.freq)
        self._last_phase, self._last_t = phase, t_last

        self._update_state(now)
        self._schedule(phase, env, t_last, now, earliest)

    def _schedule(self, phase, env, t_last, now, earliest):
        f = self.freq
        period = 1.0 / f
        gated = self.s.gate > 0 and len(self.envs) > 20 and env < self.s.gate * np.median(self.envs)
        with self.lock:
            # Don't touch a pulse that is about to play.
            if self.pending and self.pending.t < earliest:
                return
            if not self.enabled or self.state.asleep or gated:
                self.pending = None
                return
            target = self.onset_phase(f)
            t = t_last + ((target - phase) % TWO_PI) / (TWO_PI * f)
            while t < earliest or t < self.last_onset + self.s.min_gap_cycles * period:
                t += period
            self.pending = Pulse(t=t, freq=f, env=env, duration=self.s.pulse_cycles * period)

    def take_pulse(self, t_end: float) -> Pulse | None:
        """Audio side: claim the pending pulse if it starts before t_end."""
        with self.lock:
            p = self.pending
            if p is not None and p.t < t_end:
                self.pending = None
                self.last_onset = p.t
                return p
        return None

    # ------------------------------------------------------------------ state
    def _update_state(self, now: float):
        if self._last_state_t is not None and now - self._last_state_t < 1.0:
            return
        self._last_state_t = now
        need = int(self.s.fs * 4)
        if len(self.buf) < need:
            return
        x = np.fromiter(self.buf, float, len(self.buf))[-need:]
        if np.ptp(x) > 400:          # blink / movement artefact: skip this second
            return
        bp = band_powers(x, self.s.fs)
        st = self.state
        if len(self._baseline) < 90:  # first ~90 clean seconds = baseline
            self._baseline.append(bp)
        base_a = np.median([b["alpha"] for b in self._baseline]) + 1e-12
        base_ta = np.median([b["theta"] / (b["alpha"] + 1e-12) for b in self._baseline])
        ratio = bp["alpha"] / (bp["alpha"] + bp["beta"] + 1e-12)
        st.relax = 0.9 * st.relax + 0.1 * float(np.clip((ratio - 0.2) / 0.6, 0, 1))
        st.alpha_rel = 0.8 * st.alpha_rel + 0.2 * bp["alpha"] / base_a
        ta = bp["theta"] / (bp["alpha"] + 1e-12)
        st.drowsy = 0.9 * st.drowsy + 0.1 * float(np.clip((ta / (base_ta + 1e-12) - 1) / 1.5, 0, 1))
        st.history.append((now, st.relax, st.drowsy, st.alpha_rel))
        # Sleep onset heuristic: alpha has collapsed and theta dominates for ~90 s.
        recent = [h for h in st.history if now - h[0] < 90]
        if len(self._baseline) >= 90 and len(recent) > 80:
            sleepy = sum(1 for h in recent if h[3] < 0.65 and h[2] > 0.5)
            if sleepy >= 0.75 * len(recent):
                st.asleep = True
