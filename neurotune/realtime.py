"""Live mode: Muse (via LSL) -> Brain -> Synth -> your sound card."""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from .brain import Brain, Settings
from .synth import Synth

CONFIG = Path("neurotune_config.json")
CHANNELS = ["TP9", "AF7", "AF8", "TP10"]


def load_config() -> dict:
    return json.loads(CONFIG.read_text()) if CONFIG.exists() else {}


def save_config(**kw):
    cfg = load_config()
    cfg.update(kw)
    CONFIG.write_text(json.dumps(cfg, indent=2))
    print(f"Saved to {CONFIG.resolve()}: {kw}")


def open_eeg(timeout=10.0):
    """Find the Muse EEG stream on LSL (BlueMuse, OpenMuse, muselsl or the simulator)."""
    from pylsl import StreamInlet, proc_clocksync, proc_dejitter, resolve_byprop
    print("Looking for an EEG stream on LSL...")
    streams = resolve_byprop("type", "EEG", timeout=timeout)
    if not streams:
        raise SystemExit("No EEG stream found. Start BlueMuse / OpenMuse / `python -m neurotune simulate` first.")
    info = streams[0]
    inlet = StreamInlet(info, max_buflen=10, processing_flags=proc_clocksync | proc_dejitter)
    labels = []
    ch = inlet.info().desc().child("channels").child("channel")
    for _ in range(info.channel_count()):
        labels.append(ch.child_value("label") or f"ch{len(labels)}")
        ch = ch.next_sibling()
    print(f"Connected to '{info.name()}' @ {info.nominal_srate():.0f} Hz, channels {labels}")
    return inlet, labels, info.nominal_srate()


class ClockAlign:
    """Put stream timestamps on pylsl.local_clock().

    Most LSL streamers stamp samples with local_clock(), but BlueMuse defaults
    to Unix time, which would schedule every pulse decades away. If the newest
    stamp is more than 1 s off the local clock, subtract the offset, estimated
    as the max of (newest stamp - now) over ~10 s so the least-delayed chunk
    sets it (same meaning as a native stamp: roughly arrival time).
    """

    def __init__(self):
        self.active = None
        self.recent = deque(maxlen=200)

    def __call__(self, ts: np.ndarray, now: float) -> np.ndarray:
        d = float(ts[-1]) - now
        if self.active is None:
            self.active = abs(d) > 1.0
            if self.active:
                print(f"Stream timestamps are {d:.0f} s off the LSL clock (BlueMuse Unix time?) - correcting.")
        if not self.active:
            return ts
        self.recent.append(d)
        return ts - max(self.recent)


def save_session(rows, labels, synth, brain, s, ch, mode):
    out = Path("sessions")
    out.mkdir(exist_ok=True)
    f = out / time.strftime(f"%Y%m%d-%H%M%S_{mode}.npz")
    np.savez_compressed(f, eeg=np.array(rows), channels=np.array(labels[:4]),
                        pulses=np.array([[p["t"], p["played"], p["freq"]] for p in synth.log]),
                        fs=s.fs, iaf=s.iaf, target=s.target, p1=s.p1_latency_s, channel=labels[ch],
                        eeg_latency=s.eeg_latency_s,
                        state=np.array(brain.state.history))
    return f


def _channel_index(labels, name):
    for i, lab in enumerate(labels):
        if lab.upper().startswith(name.upper()):
            return i
    return CHANNELS.index(name) if name in CHANNELS else 0


def record(seconds: float):
    inlet, labels, fs = open_eeg()
    rows, t_end = [], time.time() + seconds
    while time.time() < t_end:
        data, ts = inlet.pull_chunk(timeout=0.2)
        rows += [(t, *d) for d, t in zip(data, ts)]
    return np.array(rows), labels, fs


def calibrate(seconds: float = 120):
    """Eyes-closed rest -> individual alpha frequency and best channel."""
    from .dsp import find_iaf
    print(f"Sit still, eyes CLOSED, relaxed, for {seconds:.0f} s...")
    eeg, labels, fs = record(seconds)
    best = None
    for i, lab in enumerate(labels[:4]):
        x = eeg[:, i + 1]
        x = x[np.abs(x - np.median(x)) < 200]          # drop gross artefacts
        iaf, snr = find_iaf(x, fs)
        print(f"  {lab:5s} alpha peak {iaf:5.2f} Hz  strength x{snr:4.1f}")
        if best is None or snr > best[2]:
            best = (lab, iaf, snr)
    lab, iaf, snr = best
    if snr < 2:
        print("Alpha is weak. Wet the sensors / adjust the band and try again.")
    save_config(iaf=round(iaf, 2), channel=lab, fs=fs)


def run(mode="sleep", minutes=30.0, volume_db=-12.0, target="trough", device=None,
        blocksize=256, save=True, gate=None, osc=None):
    import sounddevice as sd
    from pylsl import local_clock

    device = int(device) if isinstance(device, str) and device.isdigit() else device
    cfg = load_config()
    inlet, labels, fs = open_eeg()
    s = Settings(fs=fs, iaf=cfg.get("iaf", 10.0), target=target,
                 eeg_latency_s=cfg.get("eeg_latency_ms", 30.0) / 1000,
                 phase_bias_deg=cfg.get("phase_bias_deg", 0.0))
    if gate is not None:
        s.gate = gate
    ch_name = cfg.get("channel", "TP9")
    ch = _channel_index(labels, ch_name)
    if "eeg_latency_ms" not in cfg:
        print("! No measured EEG latency yet - assuming 30 ms. Run `loopback` for real accuracy.")
    brain = Brain(s)
    sr = 48000
    synth = Synth(brain, sr=sr, mode=mode, volume_db=volume_db)
    rows = []
    stop = threading.Event()
    lead = {"s": 0.05}                      # updated from the audio callback
    align = ClockAlign()
    client = None
    if osc:                                 # e.g. "127.0.0.1:9000" -> Ableton (Max for Live), TouchDesigner
        from pythonosc.udp_client import SimpleUDPClient
        host, port = osc.split(":")
        client = SimpleUDPClient(host, int(port))
        print(f"Sending OSC to {osc}: /neurotune/pulse <secs_until_onset> <alpha_hz>, /neurotune/state <relax> <drowsy> <alpha_rel>")

    def eeg_loop():
        while not stop.is_set():
            data, ts = inlet.pull_chunk(timeout=0.05)
            if not ts:
                continue
            arr = np.asarray(data)
            now = local_clock()
            ts = align(np.asarray(ts), now)
            brain.feed(arr[:, ch], ts, now=now, earliest=now + lead["s"] + 0.003)
            rows.extend((t, *d[:4]) for d, t in zip(data, ts))

    def callback(outdata, frames, tinfo, status):
        now = local_clock()
        dac = now + max(0.0, tinfo.outputBufferDacTime - tinfo.currentTime)
        lead["s"] = dac - now + frames / sr
        before = len(synth.log)
        outdata[:] = synth.render(frames, dac)
        if client:
            for p in synth.log[before:]:
                if p["played"]:
                    client.send_message("/neurotune/pulse", [float(p["t"] - now), float(p["freq"])])

    threading.Thread(target=eeg_loop, daemon=True).start()
    print(f"Mode: {mode} | alpha {s.iaf} Hz on {ch_name} | target P1 at {target} | "
          f"EEG latency {s.eeg_latency_s*1000:.0f} ms. Ctrl+C to stop.")
    print("Use WIRED headphones or speakers: Bluetooth audio adds 150-300 ms of shifting delay.")
    t0 = time.time()
    try:
        with sd.OutputStream(samplerate=sr, channels=2, blocksize=blocksize, dtype="float32",
                             latency="low", device=device, callback=callback):
            while time.time() - t0 < minutes * 60:
                time.sleep(1)
                st = brain.state
                if client:
                    client.send_message("/neurotune/state", [float(st.relax), float(st.drowsy), float(st.alpha_rel)])
                n = sum(1 for p in synth.log if p["played"])
                print(f"\r{time.time()-t0:6.0f}s  pulses {n:6d}  alpha {brain.freq:5.2f} Hz  "
                      f"relax {st.relax:.2f}  drowsy {st.drowsy:.2f}  {'ASLEEP' if st.asleep else ''}   ",
                      end="", flush=True)
                if st.asleep and synth.master < 0.01:
                    print("\nSleep onset detected - faded out.")
                    break
            synth.fading = True
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
    if save and rows:
        f = save_session(rows, labels, synth, brain, s, ch, mode)
        print(f"\nSaved {f}. Check accuracy with: python -m neurotune analyze {f}")


def loopback(n_clicks=40, device=None):
    """Measure the real timestamp delay of your Muse + PC chain (see README wiring)."""
    import sounddevice as sd
    from pylsl import local_clock
    device = int(device) if isinstance(device, str) and device.isdigit() else device
    inlet, labels, fs = open_eeg()
    sr = 48000
    rng = np.random.default_rng()
    times = np.cumsum(rng.uniform(1.2, 2.0, n_clicks)) + 2.0
    clicks, rows = [], []
    state = {"i": 0, "t0": None}
    click = np.zeros(int(0.004 * sr)) + 0.9
    align = ClockAlign()

    def callback(outdata, frames, tinfo, status):
        now = local_clock()
        dac = now + max(0.0, tinfo.outputBufferDacTime - tinfo.currentTime)
        if state["t0"] is None:
            state["t0"] = dac
        outdata[:] = 0
        i = state["i"]
        if i < n_clicks:
            idx = int(round((state["t0"] + times[i] - dac) * sr))
            if idx < frames:
                idx = max(idx, 0)
                m = min(frames - idx, len(click))
                outdata[idx:idx + m, 0] = click[:m]
                clicks.append(dac + idx / sr)
                state["i"] += 1

    print(f"Playing {n_clicks} clicks into the electrode wire (~{times[-1]:.0f} s)...")
    with sd.OutputStream(samplerate=sr, channels=2, blocksize=128, latency="low",
                         device=device, callback=callback):
        end = time.time() + times[-1] + 2
        while time.time() < end:
            data, ts = inlet.pull_chunk(timeout=0.2)
            if ts:
                ts = align(np.asarray(ts), local_clock())
            rows += [(t, *d[:4]) for d, t in zip(data, ts)]
    Path("sessions").mkdir(exist_ok=True)
    f = Path("sessions") / "loopback.npz"
    np.savez_compressed(f, eeg=np.array(rows), clicks=np.array(clicks), fs=fs)
    from .analyze import analyze_loopback
    r = analyze_loopback(str(f))
    print(r)
    if r["eeg_latency_ms"] is not None and r["detected"] >= n_clicks // 2:
        save_config(eeg_latency_ms=r["eeg_latency_ms"])
    else:
        print("Too few clicks detected - check the wiring and volume, then retry.")
