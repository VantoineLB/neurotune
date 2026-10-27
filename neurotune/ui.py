"""Control panel: a live session with sliders for every sound layer, sensor
contact, and brain-state meters.

  python -m neurotune ui

Everything changes while the sound plays; the session is saved to sessions/
when you press Stop or close the window.
"""
from __future__ import annotations

import threading
import time
import tkinter as tk
from collections import deque
from tkinter import ttk

import numpy as np

from .brain import Brain, Settings
from .dsp import ECHT
from .realtime import CHANNELS, CONFIG, ClockAlign, _channel_index, load_config, open_eeg, save_session
from .scope import LaserScope
from .synth import Synth

SR = 48000
BG, PANEL, FG, DIM, ACCENT, GOOD, WARN, BAD = (
    "#15171c", "#1f232b", "#e6e8ec", "#8a93a3", "#7aa2f7", "#5fd38d", "#e5c07b", "#ef6b73")


class Engine:
    """Muse -> Brain -> Synth -> sound card, with knobs the UI can turn."""

    def __init__(self, mode="music"):
        cfg = load_config()
        self.inlet, self.labels, fs = open_eeg()
        self.s = Settings(fs=fs, iaf=cfg.get("iaf", 10.0),
                          eeg_latency_s=cfg.get("eeg_latency_ms", 30.0) / 1000,
                          phase_bias_deg=cfg.get("phase_bias_deg", 0.0))
        self.brain = Brain(self.s)
        self.synth = Synth(self.brain, sr=SR, mode=mode, volume_db=-6.0)
        self.ch = _channel_index(self.labels, cfg.get("channel", "TP9"))
        self.raw = deque(maxlen=int(fs * 2))     # last 2 s, all channels, for contact check
        self.rows = []
        self.lead = {"s": 0.05}
        # Extra delay after our output (e.g. VB-Cable -> Ableton monitoring): sounds are
        # scheduled as if they leave this much later, so they still land on target.
        self.extra_latency_s = cfg.get("ableton_latency_ms", 0.0) / 1000
        self.scope = np.zeros(4096, dtype=np.float32)   # last rendered audio, for the visualizer
        self.align = ClockAlign()
        self.stream = None
        self.stop_evt = threading.Event()
        self.t0 = time.time()
        threading.Thread(target=self._eeg_loop, daemon=True).start()

    def _eeg_loop(self):
        from pylsl import local_clock
        while not self.stop_evt.is_set():
            data, ts = self.inlet.pull_chunk(timeout=0.05)
            if not ts:
                continue
            arr = np.asarray(data)
            now = local_clock()
            ts = self.align(np.asarray(ts), now)
            self.raw.extend(arr[:, :4])
            self.brain.feed(arr[:, self.ch], ts, now=now, earliest=now + self.lead["s"] + 0.003)
            self.rows.extend((t, *d[:4]) for d, t in zip(data, ts))

    def _callback(self, outdata, frames, tinfo, status):
        from pylsl import local_clock
        now = local_clock()
        dac = now + max(0.0, tinfo.outputBufferDacTime - tinfo.currentTime) + self.extra_latency_s
        self.lead["s"] = dac - now + frames / SR
        outdata[:] = self.synth.render(frames, dac)
        self.scope = np.concatenate([self.scope[frames:], outdata[:, 0]])

    def start_audio(self, device=None):
        import sounddevice as sd
        self.stop_audio()
        self.stream = sd.OutputStream(samplerate=SR, channels=2, blocksize=256, dtype="float32",
                                      latency="low", device=device, callback=self._callback)
        self.stream.start()

    def stop_audio(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _reset_tracker(self):
        b = self.brain
        with b.lock:
            b.buf.clear(); b.tbuf.clear(); b.envs.clear()
            b._last_phase = None
            b.pending = None

    def set_channel(self, name):
        self.ch = _channel_index(self.labels, name)
        self._reset_tracker()

    def set_iaf(self, iaf):
        s = self.s
        s.iaf = iaf
        self.brain.echt = ECHT(s.fs, self.brain.n, (iaf - s.band_halfwidth, iaf + s.band_halfwidth))
        self.brain.freq = iaf
        self._reset_tracker()

    def alpha_wave(self, seconds=1.5):
        """Alpha-band EEG of the tracked sensor (newest last), for the visualizer."""
        from scipy import signal
        n = int(self.s.fs * seconds)
        if len(self.raw) < n + 64:
            return None
        x = np.asarray(self.raw)[:, self.ch]
        sos = signal.butter(2, (self.s.iaf - 2, self.s.iaf + 2), "bandpass", fs=self.s.fs, output="sos")
        return signal.sosfiltfilt(sos, x - x.mean())[-n:]

    def contact(self):
        """Per-channel (label, status, std) from the last 2 s of raw EEG."""
        if len(self.raw) < 64:
            return []
        x = np.asarray(self.raw)
        out = []
        for i, lab in enumerate(self.labels[:4]):
            ptp, sd_ = np.ptp(x[:, i]), np.std(x[:, i])
            status = "no contact" if ptp > 1500 else "noisy" if sd_ > 80 else "good"
            out.append((lab, status, sd_))
        return out

    def alpha_strength(self):
        e = self.brain.envs
        if len(e) < 20:
            return 0.0
        return float(e[-1] / (np.median(e) + 1e-12))

    def close(self):
        self.synth.fading = True
        time.sleep(0.3)
        self.stop_audio()
        self.stop_evt.set()
        if self.rows:
            return save_session(self.rows, self.labels, self.synth, self.brain, self.s, self.ch,
                                self.synth.mode)


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.eng: Engine | None = None
        self.last_pulses = 0
        self.flash = 0
        cfg = load_config()
        # Values applied when a session starts (sliders keep these up to date).
        self._pending = {"volume_db": -6.0}
        self._pending_mix = {"pad_octave": 1}
        self._gate, self._iaf = 0.8, cfg.get("iaf", 10.0)
        self._latency_ms = cfg.get("ableton_latency_ms", 0.0)
        self.scope = None
        root.title("neurotune")
        root.configure(bg=BG)
        root.minsize(900, 560)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=PANEL, background=PANEL, foreground=FG)

        top = tk.Frame(root, bg=BG)
        top.pack(fill="x", padx=16, pady=(14, 6))
        self.btn = tk.Button(top, text="Start", width=10, command=self.toggle, bg=ACCENT, fg=BG,
                             font=("Segoe UI", 11, "bold"), relief="flat", activebackground=FG)
        self.btn.pack(side="left")
        tk.Button(top, text="Test tone", command=self.test_tone, bg=PANEL, fg=FG, relief="flat",
                  activebackground=ACCENT, font=("Segoe UI", 10)).pack(side="left", padx=8)
        tk.Button(top, text="Laser scope", command=self.open_scope, bg=PANEL, fg=ACCENT, relief="flat",
                  activebackground=ACCENT, font=("Segoe UI", 10)).pack(side="left")
        self.status = tk.Label(top, text="Start BlueMuse streaming, then press Start.", bg=BG, fg=DIM,
                               font=("Segoe UI", 10))
        self.status.pack(side="left", padx=8)

        dev = tk.Frame(root, bg=BG)
        dev.pack(fill="x", padx=16, pady=4)
        tk.Label(dev, text="Output", bg=BG, fg=DIM, font=("Segoe UI", 10)).pack(side="left")
        self.devices = self._output_devices()
        self.dev_var = tk.StringVar(value=self._default_device_label())
        cb = ttk.Combobox(dev, textvariable=self.dev_var, values=[d[1] for d in self.devices],
                          state="readonly", width=60)
        cb.pack(side="left", padx=8)
        cb.bind("<<ComboboxSelected>>", lambda e: self.change_device())

        body = tk.Frame(root, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=8)
        sound = self._panel(body, "Sound")
        sound.pack(side="left", fill="both", expand=True, padx=(0, 8))
        brain = self._panel(body, "Brain")
        brain.pack(side="left", fill="both", expand=True)

        # --- sound controls -------------------------------------------------
        self.mode = tk.StringVar(value="music")
        self._radio(sound, "Mode", self.mode, ["music", "sleep"], self.change_mode)
        self._slider(sound, "Master volume (dB)", -40, 0, -6, 1, lambda v: self._set("volume_db", v))
        self._slider(sound, "Pad level (dB)", -30, 12, 0, 1, lambda v: self._mix("pad_db", v))
        self._slider(sound, "Pad octave", 0, 3, 1, 1, lambda v: self._mix("pad_octave", int(v)))
        self._slider(sound, "Pad brightness", 0.2, 3, 1, 0.1, lambda v: self._mix("brightness", v))
        self._slider(sound, "Breaths / min", 2, 12, 6, 0.5, lambda v: self._mix("breath_bpm", v))
        self._slider(sound, "Seconds per chord", 4, 40, 20, 1, lambda v: self._mix("chord_seconds", v))
        self._slider(sound, "Shimmer level (dB)", -30, 12, 0, 1, lambda v: self._mix("shimmer_db", v))
        self._slider(sound, "Shimmer octave", -2, 1, 0, 1, lambda v: self._mix("shimmer_octave", int(v)))
        self._slider(sound, "Shimmer ring (s)", 0.02, 1.2, 0.35, 0.01, lambda v: self._mix("shimmer_decay", v))
        self._slider(sound, "Rain / noise (dB)", -40, 12, 0, 1, lambda v: self._mix("rain_db", v))
        self._slider(sound, "Reverb", 0, 1, 0.35, 0.05, lambda v: self._mix("reverb", v))
        self._slider(sound, "Ableton latency (ms)", 0, 200,
                     self._latency_ms, 1, self.change_latency)

        # --- brain controls + meters ---------------------------------------
        self.contact_lbls = {}
        cf = tk.Frame(brain, bg=PANEL)
        cf.pack(fill="x", pady=(0, 6))
        for name in CHANNELS:
            lb = tk.Label(cf, text=f"{name}\n–", bg=PANEL, fg=DIM, width=10, font=("Segoe UI", 9))
            lb.pack(side="left", expand=True)
            self.contact_lbls[name] = lb
        self.chan = tk.StringVar(value=cfg.get("channel", "TP9"))
        self._radio(brain, "Track sensor", self.chan, CHANNELS, self.change_channel)
        self.target = tk.StringVar(value="trough")
        self._radio(brain, "Aim at", self.target, ["trough", "peak"], self.change_target)
        self.pulses_on = tk.BooleanVar(value=True)
        tk.Checkbutton(brain, text="Brain-locked pulses", variable=self.pulses_on, command=self.change_pulses,
                       bg=PANEL, fg=FG, selectcolor=BG, activebackground=PANEL, activeforeground=FG,
                       font=("Segoe UI", 10)).pack(anchor="w", pady=2)
        self._slider(brain, "Alpha gate (0 = every cycle)", 0, 1.5, 0.8, 0.05, self.change_gate)
        self._slider(brain, "Alpha frequency (Hz)", 7, 13, cfg.get("iaf", 10.0), 0.1, self.change_iaf)

        self.meters = {}
        for key, label, color in (("strength", "Alpha strength", ACCENT), ("relax", "Relax", GOOD),
                                  ("drowsy", "Drowsy", WARN)):
            self.meters[key] = self._meter(brain, label, color)
        info = tk.Frame(brain, bg=PANEL)
        info.pack(fill="x", pady=(8, 0))
        self.pulse_dot = tk.Canvas(info, width=22, height=22, bg=PANEL, highlightthickness=0)
        self.pulse_dot.pack(side="left")
        self.dot = self.pulse_dot.create_oval(3, 3, 19, 19, fill=BG, outline=DIM)
        self.info = tk.Label(info, text="", bg=PANEL, fg=FG, font=("Consolas", 10), justify="left")
        self.info.pack(side="left", padx=6)

        root.protocol("WM_DELETE_WINDOW", self.quit)
        self.tick()

    # ---------------------------------------------------------------- widgets
    def _panel(self, parent, title):
        f = tk.Frame(parent, bg=PANEL, padx=12, pady=10)
        tk.Label(f, text=title, bg=PANEL, fg=ACCENT, font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 6))
        return f

    def _slider(self, parent, label, lo, hi, init, res, cb):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x")
        tk.Label(row, text=label, bg=PANEL, fg=DIM, font=("Segoe UI", 9), width=19, anchor="w",
                 justify="left", wraplength=150).pack(side="left")
        sc = tk.Scale(row, from_=lo, to=hi, resolution=res, orient="horizontal", showvalue=True,
                      command=lambda v: cb(float(v)), bg=PANEL, fg=FG, troughcolor=BG,
                      highlightthickness=0, activebackground=ACCENT, sliderrelief="flat",
                      font=("Segoe UI", 8), length=200, width=10)
        sc.set(init)
        sc.pack(side="left", fill="x", expand=True)
        return sc

    def _radio(self, parent, label, var, options, cb):
        f = tk.Frame(parent, bg=PANEL)
        f.pack(fill="x", pady=3)
        tk.Label(f, text=label, bg=PANEL, fg=DIM, font=("Segoe UI", 9), width=12, anchor="w").pack(side="left")
        for o in options:
            tk.Radiobutton(f, text=o, value=o, variable=var, command=cb, bg=PANEL, fg=FG, selectcolor=BG,
                           activebackground=PANEL, activeforeground=FG, font=("Segoe UI", 10)).pack(side="left")

    def _meter(self, parent, label, color):
        f = tk.Frame(parent, bg=PANEL)
        f.pack(fill="x", pady=3)
        tk.Label(f, text=label, bg=PANEL, fg=DIM, font=("Segoe UI", 9), width=14, anchor="w").pack(side="left")
        c = tk.Canvas(f, height=14, bg=BG, highlightthickness=0)
        c.pack(side="left", fill="x", expand=True)
        bar = c.create_rectangle(0, 0, 0, 14, fill=color, width=0)
        mark = c.create_line(0, 0, 0, 14, fill=FG)
        return c, bar, mark

    # ---------------------------------------------------------------- devices
    def _output_devices(self):
        import sounddevice as sd
        apis = sd.query_hostapis()
        out = []
        for i, d in enumerate(sd.query_devices()):
            api = apis[d["hostapi"]]["name"]
            if d["max_output_channels"] > 0 and api in ("MME", "Windows WASAPI", "Windows DirectSound"):
                out.append((i, f"{i}: {d['name']}  [{api.replace('Windows ', '')}]"))
        return out

    def _default_device_label(self):
        import sounddevice as sd
        # Last choice is remembered by name (indices shift when devices come and go).
        want = load_config().get("output_device")
        for _, lab in self.devices:
            if want and lab.split(": ", 1)[1] == want:
                return lab
        idx = sd.default.device[1]
        for i, lab in self.devices:
            if i == idx:
                return lab
        return self.devices[0][1] if self.devices else ""

    def _device_index(self):
        lab = self.dev_var.get()
        return next((i for i, l in self.devices if l == lab), None)

    # ---------------------------------------------------------------- actions
    def _set(self, attr, v):
        self._pending[attr] = v
        if self.eng:
            setattr(self.eng.synth, attr, v)

    def _mix(self, key, v):
        self._pending_mix[key] = v
        if self.eng:
            self.eng.synth.mix[key] = v

    def toggle(self):
        if self.eng:
            f = self.eng.close()
            self.eng = None
            self._save_latency()
            self.btn.config(text="Start", bg=ACCENT)
            self.status.config(text=f"Stopped. Saved {f}" if f else "Stopped.")
            return
        self.status.config(text="Looking for the Muse on LSL...", fg=DIM)
        self.root.update()
        try:
            self.eng = Engine(mode=self.mode.get())
        except SystemExit as e:
            self.status.config(text=str(e), fg=BAD)
            return
        syn = self.eng.synth
        for k, v in self._pending.items():
            setattr(syn, k, v)
        syn.mix.update(self._pending_mix)
        self.change_channel(); self.change_target(); self.change_pulses()
        self.change_gate(self._gate); self.change_iaf(self._iaf); self.change_latency(self._latency_ms)
        try:
            self.eng.start_audio(self._device_index())
        except Exception as e:
            self.status.config(text=f"Audio error: {e}", fg=BAD)
        else:
            self.status.config(text=f"Connected: {self.eng.inlet.info().name()}", fg=GOOD)
        self.btn.config(text="Stop", bg=BAD)

    def test_tone(self):
        if self.eng:
            self.eng.synth.beep()
            return
        import sounddevice as sd
        t = np.arange(int(0.6 * SR)) / SR
        tone = 0.3 * np.sin(2 * np.pi * 440 * t) * np.minimum(1, np.minimum(t, t[::-1]) * 50)
        try:
            sd.play(tone.astype(np.float32), SR, device=self._device_index())
            self.status.config(text="Played a 440 Hz test tone. Heard it? If not, pick another output.", fg=DIM)
        except Exception as e:
            self.status.config(text=f"Audio error: {e}", fg=BAD)

    def change_device(self):
        import json
        cfg = load_config()
        cfg["output_device"] = self.dev_var.get().split(": ", 1)[1]
        CONFIG.write_text(json.dumps(cfg, indent=2))
        if self.eng:
            try:
                self.eng.start_audio(self._device_index())
                self.status.config(text=f"Output: {self.dev_var.get()}", fg=GOOD)
            except Exception as e:
                self.status.config(text=f"Audio error: {e}", fg=BAD)

    def change_mode(self):
        if self.eng:
            self.eng.synth.mode = self.mode.get()

    def change_channel(self):
        if self.eng:
            self.eng.set_channel(self.chan.get())

    def change_target(self):
        if self.eng:
            self.eng.s.target = self.target.get()

    def change_pulses(self):
        if self.eng:
            self.eng.brain.enabled = self.pulses_on.get()

    def change_gate(self, v):
        self._gate = v
        if self.eng:
            self.eng.s.gate = v

    def change_latency(self, v):
        self._latency_ms = v
        if self.eng:
            self.eng.extra_latency_s = v / 1000

    def _save_latency(self):
        import json
        cfg = load_config()
        if cfg.get("ableton_latency_ms") != self._latency_ms:
            cfg["ableton_latency_ms"] = self._latency_ms
            CONFIG.write_text(json.dumps(cfg, indent=2))

    def open_scope(self):
        if self.scope is None or not self.scope.alive:
            self.scope = LaserScope(self.root, lambda: self.eng)

    def change_iaf(self, v):
        self._iaf = v
        if self.eng and abs(self.eng.s.iaf - v) > 1e-6:
            self.eng.set_iaf(v)

    # ---------------------------------------------------------------- refresh
    def _bar(self, meter, frac, mark=None):
        c, bar, m = meter
        w = max(c.winfo_width(), 1)
        c.coords(bar, 0, 0, w * float(np.clip(frac, 0, 1)), 14)
        x = w * float(np.clip(mark, 0, 1)) if mark is not None else -5
        c.coords(m, x, 0, x, 14)

    def tick(self):
        e = self.eng
        if e:
            for lab, status, sd_ in e.contact():
                key = next((n for n in CHANNELS if lab.upper().startswith(n)), lab)
                if key in self.contact_lbls:
                    color = {"good": GOOD, "noisy": WARN, "no contact": BAD}[status]
                    self.contact_lbls[key].config(text=f"{key}\n{status}", fg=color)
            st = e.brain.state
            strength = e.alpha_strength()
            self._bar(self.meters["strength"], strength / 2.5, e.s.gate / 2.5)
            self._bar(self.meters["relax"], st.relax)
            self._bar(self.meters["drowsy"], st.drowsy)
            n = sum(1 for p in e.synth.log if p["played"])
            if n > self.last_pulses:
                self.flash = 3
            self.last_pulses = n
            self.pulse_dot.itemconfig(self.dot, fill=ACCENT if self.flash > 0 else BG)
            self.flash -= 1
            m, s = divmod(int(time.time() - e.t0), 60)
            self.info.config(text=f"{m:02d}:{s:02d}   pulses {n:5d}   alpha {e.brain.freq:5.2f} Hz"
                                  f"{'   ASLEEP' if st.asleep else ''}")
        self.root.after(80, self.tick)

    def quit(self):
        if self.eng:
            self.eng.close()
        self._save_latency()
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()
