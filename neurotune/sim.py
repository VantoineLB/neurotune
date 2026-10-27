"""Synthetic Muse: realistic-ish EEG with waxing/waning alpha, plus a
Bluetooth-like delivery model (12-sample packets, variable delay).

Used two ways:
  * offline  - runs the whole closed loop on a virtual clock and writes a
               WAV + logs, so the algorithm can be tested with no hardware;
  * live     - publishes the fake EEG as an LSL stream named like a Muse,
               so `run` can be tried on your PC before the headband works.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy import signal

from .brain import Brain, Settings
from .synth import Synth


class FakeEEG:
    """Generates 4-channel (TP9, AF7, AF8, TP10) EEG in microvolts."""

    def __init__(self, fs=256.0, iaf=10.0, seed=1, sleepy_after_s=None):
        self.fs, self.iaf = fs, iaf
        self.rng = np.random.default_rng(seed)
        self.phase = 0.0
        self.freq = iaf
        self.amp_state = 1.0
        self.n = 0
        self.sleepy_after_s = sleepy_after_s
        b, a = [0.049922035, -0.095993537, 0.050612699, -0.004408786], [1, -2.494956002, 2.017265875, -0.522189400]
        self._pb, self._pa = b, a
        self._zi = [signal.lfilter_zi(b, a) * 0 for _ in range(4)]
        sos = signal.butter(1, 0.5, "lowpass", fs=fs, output="sos")  # slow burst envelope
        self._env_sos, self._env_zi = sos, np.zeros((1, 2))
        self.alpha_log = []   # clean alpha (ground-truth phase) for evaluation

    def chunk(self, k: int):
        t = (self.n + np.arange(k)) / self.fs
        # frequency wanders slowly around the IAF
        self.freq += self.rng.normal(0, 0.002) * k - 0.01 * (self.freq - self.iaf)
        ph = self.phase + 2 * np.pi * self.freq * np.arange(1, k + 1) / self.fs
        self.phase = ph[-1] % (2 * np.pi)
        drive = self.rng.standard_normal(k) * 3.0
        env, self._env_zi = signal.sosfilt(self._env_sos, drive, zi=self._env_zi)
        env = np.clip(1.0 + env, 0.05, None)
        scale = 1.0
        if self.sleepy_after_s is not None and t[0] > self.sleepy_after_s:
            scale = max(0.25, 1 - (t[0] - self.sleepy_after_s) / 120)
        alpha = 12.0 * scale * env * np.cos(ph)          # ~12 uV alpha at Muse sites
        theta = (3.0 + 6.0 * (1 - scale)) * np.cos(2 * np.pi * 5.5 * t + 1.3)
        self.alpha_log.append(np.cos(ph))
        out = np.zeros((k, 4))
        for c, w in enumerate((1.0, 0.6, 0.6, 1.0)):     # alpha weaker on the forehead
            noise, self._zi[c] = signal.lfilter(self._pb, self._pa, self.rng.standard_normal(k), zi=self._zi[c])
            out[:, c] = w * alpha + theta + 400 * noise + 1.5 * np.sin(2 * np.pi * 50 * t)
        self.n += k
        return out


def run_offline(out_dir: str, minutes: float = 3.0, mode: str = "sleep", iaf: float = 10.0,
                ble_delay_ms: float = 35.0, ble_jitter_ms: float = 10.0,
                audio_latency_ms: float = 20.0, eeg_latency_ms: float | None = None,
                channel: str = "TP9", seed: int = 1, sleepy_after_s=None, settings: Settings | None = None):
    """Run the closed loop on a virtual clock. Returns a summary dict."""
    fs, sr, block = 256.0, 48000, 256
    s = settings or Settings(fs=fs, iaf=iaf)
    # EEG timestamps arrive late by roughly the BLE delay (as with LSL dejitter on
    # arrival times). eeg_latency_ms tells the brain how much to correct for.
    s.eeg_latency_s = (ble_delay_ms if eeg_latency_ms is None else eeg_latency_ms) / 1000
    brain = Brain(s)
    synth = Synth(brain, sr=sr, mode=mode, seed=seed)
    gen = FakeEEG(fs, iaf, seed, sleepy_after_s)
    rng = np.random.default_rng(seed + 7)
    ch = ["TP9", "AF7", "AF8", "TP10"].index(channel)

    total = int(minutes * 60 * sr / block)
    audio, eeg_rows, packets = [], [], []
    next_pkt_t = 0.0
    lat = audio_latency_ms / 1000
    last_arrival = 0.0
    for b in range(total):
        now = b * block / sr
        # generate 12-sample packets up to `now`, give each a delivery time
        while next_pkt_t <= now:
            data = gen.chunk(12)
            t_true = next_pkt_t + np.arange(12) / fs
            arrival = max(last_arrival + 0.001, t_true[-1] + (ble_delay_ms + rng.normal(0, ble_jitter_ms)) / 1000)
            last_arrival = arrival
            packets.append((arrival, data, t_true + ble_delay_ms / 1000))  # stamped late, like LSL
            next_pkt_t += 12 / fs
        while packets and packets[0][0] <= now:
            _, data, stamps = packets.pop(0)
            brain.feed(data[:, ch], stamps, now=now, earliest=now + lat + block / sr + 0.003)
            for r, t in zip(data, stamps):
                eeg_rows.append((t, *r))
        audio.append(synth.render(block, now + lat))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    sf.write(out / f"session_{mode}.wav", np.concatenate(audio), sr)
    eeg = np.array(eeg_rows)
    np.savez_compressed(out / f"session_{mode}.npz", eeg=eeg, channels=np.array(["TP9", "AF7", "AF8", "TP10"]),
                        pulses=np.array([[p["t"], p["played"], p["freq"]] for p in synth.log]),
                        truth=np.concatenate(gen.alpha_log), fs=fs, iaf=iaf,
                        target=s.target, p1=s.p1_latency_s, channel=channel,
                        ble_delay=ble_delay_ms / 1000)
    played = [p for p in synth.log if p["played"]]
    summary = {"mode": mode, "minutes": minutes, "pulses": len(played),
               "pulses_per_s": len(played) / (minutes * 60),
               "dropped_late": sum(1 for p in synth.log if not p["played"]),
               "asleep_detected": brain.state.asleep}
    (out / f"session_{mode}.json").write_text(json.dumps(summary, indent=2))
    return summary


def stream_lsl(iaf: float = 10.0, seed: int = 1):
    """Publish fake EEG as an LSL stream so `run` can be tested without a headband."""
    from pylsl import StreamInfo, StreamOutlet, local_clock
    info = StreamInfo("Muse-SIM", "EEG", 4, 256, "float32", "neurotune-sim")
    chans = info.desc().append_child("channels")
    for lab in ("TP9", "AF7", "AF8", "TP10"):
        chans.append_child("channel").append_child_value("label", lab)
    outlet = StreamOutlet(info, chunk_size=12)
    gen = FakeEEG(256.0, iaf, seed)
    print(f"Streaming fake Muse EEG (alpha at {iaf} Hz) on LSL as 'Muse-SIM'. Ctrl+C to stop.")
    t0 = local_clock()
    sent = 0
    while True:
        due = int((local_clock() - t0) * 256)
        while sent + 12 <= due:
            data = gen.chunk(12)
            stamp = t0 + (sent + 11) / 256
            outlet.push_chunk(data.astype(np.float32).tolist(), stamp)
            sent += 12
        time.sleep(0.01)
