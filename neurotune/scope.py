"""Laser scope: a floating, borderless window with a glowing blue line.

Brain view: your live alpha wave (the rhythm the pulses lock to), flaring on
every brain-locked pulse. Sound view: the audio neurotune is playing.

  click          brain / sound view
  drag           move
  mouse wheel    resize
  double-click   fullscreen
  right-click    menu (always on top, close)
"""
from __future__ import annotations

import time
import tkinter as tk

import numpy as np

BLACK = "#02040a"
# Glow layers, outermost first: (width, base colour, flare colour)
LAYERS = [
    (26, (3, 10, 32), (20, 40, 110)),
    (16, (6, 30, 90), (40, 90, 220)),
    (9, (20, 90, 230), (90, 170, 255)),
    (4, (90, 175, 255), (190, 230, 255)),
    (1.6, (225, 245, 255), (255, 255, 255)),
]
TRAIL = 4          # ghost copies of earlier frames (phosphor persistence)
POINTS = 220


def _hex(rgb):
    return "#%02x%02x%02x" % tuple(int(np.clip(c, 0, 255)) for c in rgb)


def _mix(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


class LaserScope:
    def __init__(self, master: tk.Misc, get_engine):
        self.get_engine = get_engine
        self.win = tk.Toplevel(master)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.configure(bg=BLACK)
        self.w, self.h = 640, 220
        sx = master.winfo_screenwidth()
        self.win.geometry(f"{self.w}x{self.h}+{sx - self.w - 40}+60")
        self.c = tk.Canvas(self.win, bg=BLACK, highlightthickness=0)
        self.c.pack(fill="both", expand=True)
        self.view = "brain"
        self.gain = 1.0
        self.flare = 0.0
        self.last_pulses = 0
        self.history: list[list[float]] = []
        self.t0 = time.time()
        self.rng = np.random.default_rng()
        self.fullscreen = False
        self.on_top = tk.BooleanVar(value=True)

        self.trails = [[self.c.create_line(0, 0, 0, 0, smooth=True, capstyle="round", joinstyle="round")
                        for _ in LAYERS[1:3]] for _ in range(TRAIL)]
        self.lines = [self.c.create_line(0, 0, 0, 0, smooth=True, capstyle="round", joinstyle="round",
                                         width=w) for w, _, _ in LAYERS]
        self.label = self.c.create_text(12, 10, anchor="nw", fill="#24406e", font=("Segoe UI", 8))

        self.menu = tk.Menu(self.win, tearoff=0)
        self.menu.add_command(label="Brain / sound view", command=self.toggle_view)
        self.menu.add_checkbutton(label="Always on top", variable=self.on_top,
                                  command=lambda: self.win.attributes("-topmost", self.on_top.get()))
        self.menu.add_separator()
        self.menu.add_command(label="Close", command=self.close)

        self._drag = None
        self.c.bind("<ButtonPress-1>", self._press)
        self.c.bind("<B1-Motion>", self._move)
        self.c.bind("<ButtonRelease-1>", self._release)
        self.c.bind("<Double-Button-1>", self._fullscreen)
        self.c.bind("<MouseWheel>", self._wheel)
        self.c.bind("<Button-3>", lambda e: self.menu.tk_popup(e.x_root, e.y_root))
        self.win.bind("<Escape>", lambda e: self._fullscreen() if self.fullscreen else None)
        self.alive = True
        self.frame()

    # --------------------------------------------------------------- input
    def _press(self, e):
        self._drag = (e.x_root, e.y_root, self.win.winfo_x(), self.win.winfo_y(), False)

    def _move(self, e):
        if self._drag and not self.fullscreen:
            x0, y0, wx, wy, _ = self._drag
            dx, dy = e.x_root - x0, e.y_root - y0
            if abs(dx) + abs(dy) > 3:
                self._drag = (x0, y0, wx, wy, True)
                self.win.geometry(f"+{wx + dx}+{wy + dy}")

    def _release(self, e):
        if self._drag and not self._drag[4]:
            self.toggle_view()
        self._drag = None

    def _wheel(self, e):
        if self.fullscreen:
            return
        k = 1.1 if e.delta > 0 else 1 / 1.1
        self.w = int(np.clip(self.w * k, 240, 2400))
        self.h = int(np.clip(self.h * k, 90, 900))
        self.win.geometry(f"{self.w}x{self.h}")

    def _fullscreen(self, e=None):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self._saved = self.win.geometry()
            self.win.geometry(f"{self.win.winfo_screenwidth()}x{self.win.winfo_screenheight()}+0+0")
        else:
            self.win.geometry(self._saved)

    def toggle_view(self):
        self.view = "sound" if self.view == "brain" else "brain"
        self.history.clear()

    def close(self):
        self.alive = False
        self.win.destroy()

    # --------------------------------------------------------------- signal
    def _signal(self, eng):
        t = time.time() - self.t0
        if eng is None:            # idle: a slow breathing sine so it's never dead
            x = np.linspace(0, 1, POINTS)
            return 0.35 * np.sin(2 * np.pi * (3 * x - 0.4 * t)) * (0.6 + 0.4 * np.sin(0.6 * t))
        if self.view == "sound":
            a = eng.scope
            # trigger on a rising zero crossing so the wave stands still
            seg = a[-3000:]
            zc = np.where((seg[:-1] <= 0) & (seg[1:] > 0))[0]
            start = zc[0] if len(zc) else 0
            y = seg[start:start + 1600]
            if len(y) < 1600:
                y = a[-1600:]
            y = y[:: max(1, len(y) // POINTS)][:POINTS]
        else:
            y = eng.alpha_wave(0.6)     # ~6 cycles: readable bends, not a comb
            if y is None:
                return np.zeros(POINTS)
            y = np.interp(np.linspace(0, len(y) - 1, POINTS), np.arange(len(y)), y)
        # slow auto-gain: fills the window but still lets bursts swell
        peak = np.percentile(np.abs(y), 95) + 1e-9
        self.gain += 0.03 * (0.3 / peak - self.gain)
        return 0.85 * np.tanh(y * self.gain / 0.85)   # soft-limit: bursts bend, never flat-top

    # --------------------------------------------------------------- render
    def frame(self):
        if not self.alive:
            return
        eng = self.get_engine()
        W, H = max(self.c.winfo_width(), 10), max(self.c.winfo_height(), 10)
        y = self._signal(eng)

        if eng is not None:
            n = sum(1 for p in eng.synth.log if p["played"])
            if n > self.last_pulses:
                self.flare = 1.0
            self.last_pulses = n
        f = self.flare
        # electric shimmer: fine jitter, stronger while flaring
        y = y + self.rng.normal(0, 0.006 + 0.03 * f, len(y))
        xs = np.linspace(W * 0.03, W * 0.97, len(y))
        ys = H / 2 - y * H / 2
        coords = np.column_stack([xs, ys]).ravel().tolist()

        self.history.insert(0, coords)
        del self.history[TRAIL + 1:]
        for k, pair in enumerate(self.trails):
            old = self.history[k + 1] if k + 1 < len(self.history) else None
            fade = 0.45 * (1 - (k + 1) / (TRAIL + 1))
            for item, (w, base, _) in zip(pair, LAYERS[1:3]):
                if old:
                    self.c.coords(item, *old)
                    self.c.itemconfig(item, width=w * 0.7, fill=_hex(_mix((2, 4, 10), base, fade)))
                else:
                    self.c.coords(item, 0, 0, 0, 0)
        for item, (w, base, hot) in zip(self.lines, LAYERS):
            self.c.coords(item, *coords)
            self.c.itemconfig(item, width=w * (1 + 0.5 * f), fill=_hex(_mix(base, hot, f)))
        self.c.tag_raise(self.label)
        if eng is None:
            text = "neurotune · press Start"
        else:
            text = (f"{'ALPHA' if self.view == 'brain' else 'SOUND'}  {eng.brain.freq:4.1f} Hz"
                    f"   relax {eng.brain.state.relax:.2f}")
        self.c.itemconfig(self.label, text=text)
        self.flare *= 0.82
        self.win.after(16, self.frame)
