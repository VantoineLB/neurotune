"""Oscilloscope: a crisp, low-latency scope of the live Muse EEG.

Reads the headband straight off LSL (nothing in between) and draws it in roll
mode, like a bench scope on a slow timebase: the newest sample is always at
the right-hand edge.

  CH1 (yellow)  one sensor's EEG, 1-40 Hz with a mains notch
  CH2 (cyan)    the same sensor's alpha wave as neurotune's tracker sees it:
                an endpoint-corrected Hilbert transform (ecHT) re-estimated at
                every new sample, so it has no phase lag at your alpha frequency

  python -m neurotune scope [--channel TP9] [--mains 50]

  F / double-click   fullscreen            1 / 2          channel on / off
  up / down          uV per division       left / right   time per division
  A                  auto-scale            space          run / stop (freeze)
  L                  smooth / live         S              save an HD still (PNG)
  Esc                leave fullscreen, quit

Latency. The Muse sends EEG in packets of 12 samples (~47 ms apart). "Live"
draws each packet the instant it arrives, so the trace steps forward ~21 times
a second. "Smooth" (default) scrolls continuously and holds back about one
packet to do it. CH1's filters add a few ms; CH2 (ecHT) adds none at your
alpha frequency. The readout at the bottom shows every term.
"""
from __future__ import annotations

import collections
import ctypes
import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline

from .dsp import ECHT
from .realtime import _channel_index, load_config

UV_STEPS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
MS_STEPS = [20, 50, 100, 200, 500, 1000]
DIV_X, DIV_Y = 10, 8
BUFFER_S = 12

BG = (5, 7, 9)
GRID = (40, 46, 52)
AXIS = (70, 78, 86)
TEXT_DIM = (110, 118, 126)
CH_COLOURS = [(255, 212, 60), (60, 214, 255)]
RUN_GREEN, STOP_RED = (70, 220, 120), (240, 80, 80)


def _dpi_aware():
    """Without this Windows renders at 96 dpi and upscales: soft, blurry lines."""
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


def _delay_ms(sos, fs, hz):
    """Group delay of a filter at one frequency, in ms."""
    b, a = signal.sos2tf(sos)
    _, gd = signal.group_delay((b, a), w=[hz], fs=fs)
    return float(gd[0]) / fs * 1000


class Channel:
    def __init__(self, sos, n):
        self.sos = sos
        self.zi = None
        self.buf = np.zeros(n)

    def push(self, x):
        if self.sos is None:
            y = x
        elif self.zi is None:
            self.zi = signal.sosfilt_zi(self.sos) * x[0]
            y, self.zi = signal.sosfilt(self.sos, x, zi=self.zi)
        else:
            y, self.zi = signal.sosfilt(self.sos, x, zi=self.zi)
        m = len(y)
        self.buf = np.concatenate([self.buf[m:], y]) if m < len(self.buf) else y[-len(self.buf):]


class Scope:
    def __init__(self, channel=None, mains=50.0):
        cfg = load_config()
        self.want_channel = channel or cfg.get("channel", "AF7")
        self.iaf = float(cfg.get("iaf", 10.0))
        self.mains = mains
        self.inlet = None
        self.name = ""
        self.fs = 256.0
        self.received = 0
        self.read_pos = 0.0
        self.chunks = collections.deque(maxlen=60)   # recent chunk sizes -> smoothing lag
        self.last_data = 0.0
        self.smooth = True
        self.running = True
        self.show = [True, True]
        self.uv = [UV_STEPS.index(50), UV_STEPS.index(20)]
        self.ms = MS_STEPS.index(100)
        self.autoscaled = False
        self.fullscreen = False
        self.last_click = 0.0
        self.next_resolve = 0.0

    # ---------------------------------------------------------------- input
    def connect(self):
        from pylsl import StreamInlet, resolve_byprop
        streams = resolve_byprop("type", "EEG", timeout=0.3)
        if not streams:
            return
        info = streams[0]
        self.inlet = StreamInlet(info, max_buflen=2, processing_flags=0)
        self.fs = info.nominal_srate() or 256.0
        labels, ch = [], self.inlet.info().desc().child("channels").child("channel")
        for _ in range(info.channel_count()):
            labels.append(ch.child_value("label") or f"ch{len(labels)}")
            ch = ch.next_sibling()
        self.ch = _channel_index(labels, self.want_channel)
        self.ch_label = labels[self.ch] if labels[self.ch] and not labels[self.ch].startswith("ch") else self.want_channel
        self.name = info.name().split(" (")[0]
        fs, n = self.fs, int(self.fs * BUFFER_S)
        wide = signal.butter(2, (1, 40), "bandpass", fs=fs, output="sos")
        if self.mains < fs / 2:
            b, a = signal.iirnotch(self.mains, 30, fs=fs)
            wide = np.vstack([wide, signal.tf2sos(b, a)])
        # same tracker settings as brain.py: 1 s window, alpha frequency +-1.5 Hz
        self.echt = ECHT(fs, int(fs), (self.iaf - 1.5, self.iaf + 1.5))
        self.raw = np.zeros(self.echt.n)
        self.chans = [Channel(wide, n), Channel(None, n)]
        self.filter_ms = [_delay_ms(wide, fs, self.iaf), 0.0]
        self.received = 0
        self.read_pos = 0.0
        self.last_good = 0.0
        self.last_data = time.perf_counter()

    def pull(self):
        if self.inlet is None:
            return
        data, _ = self.inlet.pull_chunk(timeout=0.0, max_samples=2048)
        if not data:
            return
        x = np.asarray(data, dtype=float)[:, self.ch]
        # the Muse sends NaN when a sensor loses contact: hold the last good value,
        # or one NaN would poison the filter state for good
        bad = ~np.isfinite(x)
        if bad.any():
            good = np.where(~bad, np.arange(len(x)), -1)
            good = np.maximum.accumulate(good)
            x = np.where(good >= 0, x[np.maximum(good, 0)], self.last_good)
        self.last_good = x[-1]
        self.chans[0].push(x)
        # ecHT at every new sample: one window ending on each, all FFT'd in one batch
        n, m = self.echt.n, len(x)
        ext = np.concatenate([self.raw, x])
        self.raw = ext[-n:]
        win = np.lib.stride_tricks.sliding_window_view(ext, n)[-m:]
        win = win - win.mean(axis=1, keepdims=True)
        z = np.fft.ifft(np.fft.fft(win, axis=1) * self.echt._kernel, axis=1)[:, -1]
        self.chans[1].push(np.abs(z) * np.cos(np.angle(z)))
        self.received += len(x)
        self.chunks.append(len(x))
        self.last_data = time.perf_counter()

    def lag_samples(self):
        if not self.smooth or not self.chunks:
            return 0.0
        # one typical packet (+2 samples). Sized on the median, not the max: BlueMuse
        # sometimes delivers two packets at once, and holding back for that doubles the delay.
        return float(np.median(self.chunks)) + 2.0

    def advance(self, dt):
        if not self.running or self.inlet is None:
            return
        target = self.received - self.lag_samples()
        if not self.smooth:
            self.read_pos = float(self.received)
            return
        self.read_pos += self.fs * dt
        err = target - self.read_pos
        if abs(err) > self.fs * 0.3:
            self.read_pos = target
        else:
            self.read_pos += err * min(1.0, dt * 3)
        self.read_pos = min(self.read_pos, float(self.received))

    def autoscale(self):
        if self.inlet is None or self.received < self.fs:
            return
        span = int(self.fs * MS_STEPS[self.ms] * DIV_X / 1000)
        for i, c in enumerate(self.chans):
            y = c.buf[-min(span, self.received, len(c.buf)):]
            peak = np.percentile(np.abs(y), 98)
            # biggest normal swing fills ~1.5 divisions
            want = peak / 1.5
            self.uv[i] = next((k for k, v in enumerate(UV_STEPS) if v >= want), len(UV_STEPS) - 1)
        self.autoscaled = True


def main(channel=None, mains=50.0):
    _dpi_aware()
    import os
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame

    pygame.init()
    pygame.display.set_caption("neurotune · oscilloscope")
    desktop = pygame.display.get_desktop_sizes()[0]
    windowed = (min(1600, desktop[0] - 80), min(900, desktop[1] - 120))

    vsync = {"on": False}

    def open_window(full):
        size = desktop if full else windowed
        # SCALED is what makes pygame honour vsync; with DPI awareness on and the
        # logical size equal to the window size it is still a 1:1, pixel-exact render.
        flags = (pygame.NOFRAME if full else 0) | pygame.SCALED
        if full:
            os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
        else:
            os.environ["SDL_VIDEO_CENTERED"] = "1"
            os.environ.pop("SDL_VIDEO_WINDOW_POS", None)
        try:
            surf = pygame.display.set_mode(size, flags, vsync=1)
            vsync["on"] = True
        except pygame.error:
            surf = pygame.display.set_mode(size, flags)
            vsync["on"] = False
        return surf

    def save_still(surface, name=None):
        out = Path("sessions")
        out.mkdir(exist_ok=True)
        path = out / (name or time.strftime("scope-%Y%m%d-%H%M%S.png"))
        try:
            pygame.image.save(surface, str(path))
            print(f"saved {path}", flush=True)
        except pygame.error as err:     # e.g. Dropbox has the file locked: skip, keep running
            print(f"couldn't save {path}: {err}", flush=True)

    sc = Scope(channel, mains)
    screen = open_window(False)
    layout = {}

    def build(screen):
        W, H = screen.get_size()
        k = H / 1080
        top, bottom, side = int(70 * k), int(64 * k), int(40 * k)
        plot = pygame.Rect(side, top, W - 2 * side, H - top - bottom)
        grat = pygame.Surface((W, H))
        grat.fill(BG)
        dx, dy = plot.w / DIV_X, plot.h / DIV_Y
        dot = max(1, round(1.4 * k))
        for i in range(DIV_X + 1):          # dotted division lines, 5 dots per division
            x = plot.x + i * dx
            for j in range(DIV_Y * 5 + 1):
                pygame.draw.circle(grat, GRID, (x, plot.y + j * dy / 5), dot)
        for j in range(DIV_Y + 1):
            y = plot.y + j * dy
            for i in range(DIV_X * 5 + 1):
                pygame.draw.circle(grat, GRID, (plot.x + i * dx / 5, y), dot)
        cx, cy = plot.centerx, plot.centery     # centre axes with minor ticks
        pygame.draw.line(grat, AXIS, (plot.x, cy), (plot.right, cy), 1)
        pygame.draw.line(grat, AXIS, (cx, plot.y), (cx, plot.bottom), 1)
        t = max(3, int(7 * k))
        for i in range(DIV_X * 5 + 1):
            x = plot.x + i * dx / 5
            pygame.draw.line(grat, AXIS, (x, cy - t), (x, cy + t), 1)
        for j in range(DIV_Y * 5 + 1):
            y = plot.y + j * dy / 5
            pygame.draw.line(grat, AXIS, (cx - t, y), (cx + t, y), 1)
        pygame.draw.rect(grat, AXIS, plot, 1)
        layout.update(
            W=W, H=H, k=k, plot=plot, grat=grat,
            glow=pygame.Surface((max(1, W // 4), max(1, H // 4))),
            font=pygame.font.SysFont("consolas", max(12, int(24 * k))),
            small=pygame.font.SysFont("consolas", max(11, int(19 * k))),
            width=max(1.6, 2.4 * k),
        )

    build(screen)
    clock = pygame.time.Clock()
    debug = bool(os.environ.get("NEUROTUNE_SCOPE_DEBUG"))
    stats, stats_t = [], [time.perf_counter()]
    prev = time.perf_counter()

    def trace(c, uv, offset_div, span, plot):
        """Screen points for one channel over the visible window."""
        right = sc.read_pos
        left = right - span
        start = sc.received - len(c.buf)
        i0 = max(int(np.floor(left)) - 2, start, sc.received - len(c.buf))
        i1 = min(int(np.ceil(right)) + 1, sc.received)
        if i1 - i0 < 4:
            return None
        idx = np.arange(i0, i1)
        y = np.nan_to_num(c.buf[idx - start])
        n_px = plot.w
        if len(idx) < n_px:     # fewer samples than pixels: smooth curve through them
            xs_s = np.linspace(max(left, i0), min(right, i1 - 1), n_px)
            y = CubicSpline(idx, y)(xs_s)
            idx = xs_s
        x = plot.x + (idx - left) / span * plot.w
        dy = plot.h / DIV_Y
        ys = plot.centery - (y / UV_STEPS[uv] + offset_div) * dy
        ys = np.clip(ys, plot.y, plot.bottom)
        keep = (x >= plot.x) & (x <= plot.right)
        return np.column_stack([x[keep], ys[keep]])

    while True:
        now = time.perf_counter()
        dt, prev = now - prev, now
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    if sc.fullscreen:
                        sc.fullscreen = False
                        screen = open_window(False)
                        build(screen)
                    else:
                        pygame.quit()
                        return
                elif e.key == pygame.K_f:
                    sc.fullscreen = not sc.fullscreen
                    screen = open_window(sc.fullscreen)
                    build(screen)
                elif e.key in (pygame.K_1, pygame.K_2):
                    i = e.key - pygame.K_1
                    sc.show[i] = not sc.show[i]
                elif e.key == pygame.K_UP:
                    sc.uv = [max(0, u - 1) for u in sc.uv]
                elif e.key == pygame.K_DOWN:
                    sc.uv = [min(len(UV_STEPS) - 1, u + 1) for u in sc.uv]
                elif e.key == pygame.K_LEFT:
                    sc.ms = max(0, sc.ms - 1)
                elif e.key == pygame.K_RIGHT:
                    sc.ms = min(len(MS_STEPS) - 1, sc.ms + 1)
                elif e.key == pygame.K_a:
                    sc.autoscale()
                elif e.key == pygame.K_SPACE:
                    sc.running = not sc.running
                elif e.key == pygame.K_l:
                    sc.smooth = not sc.smooth
                elif e.key == pygame.K_s:
                    save_still(screen)
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                if now - sc.last_click < 0.35:
                    sc.fullscreen = not sc.fullscreen
                    screen = open_window(sc.fullscreen)
                    build(screen)
                sc.last_click = now

        if sc.inlet is None and now >= sc.next_resolve:
            sc.connect()
            sc.next_resolve = now + 2.0
        if sc.running:
            sc.pull()
        else:
            sc.inlet and sc.inlet.flush()
        sc.advance(dt)
        if not sc.autoscaled and sc.received > sc.fs * 2:
            sc.autoscale()

        L = layout
        plot = L["plot"]
        screen.blit(L["grat"], (0, 0))
        glow = L["glow"]
        glow.fill((0, 0, 0))
        span = sc.fs * MS_STEPS[sc.ms] * DIV_X / 1000
        both = all(sc.show)
        lines = []
        if sc.inlet is not None and sc.received > 4:
            for i, c in enumerate(sc.chans):
                if not sc.show[i]:
                    continue
                off = (2 if i == 0 else -2) if both else 0
                pts = trace(c, sc.uv[i], off, span, plot)
                if pts is not None and len(pts) > 1:
                    lines.append((i, pts, off))
                    g = [(x / 4, y / 4) for x, y in pts[::2]]
                    if len(g) > 1:
                        pygame.draw.lines(glow, tuple(v // 3 for v in CH_COLOURS[i]), False, g,
                                          max(1, int(2 * L["k"])))
            screen.blit(pygame.transform.smoothscale(glow, (L["W"], L["H"])), (0, 0),
                        special_flags=pygame.BLEND_ADD)
            w = L["width"]
            for i, pts, off in lines:
                col = CH_COLOURS[i]
                for o in np.linspace(-w / 2, w / 2, max(2, int(np.ceil(w * 1.4)))):
                    pygame.draw.aalines(screen, col, False, (pts + (0, o)).tolist())
                # channel ground marker on the left edge, like a scope's "1>"
                y0 = plot.centery - off * plot.h / DIV_Y
                a = 10 * L["k"]
                pygame.draw.polygon(screen, col, [(plot.x - 2 * a, y0 - a), (plot.x - 2 * a, y0 + a),
                                                  (plot.x - 0.4 * a, y0)])

        # ---- readouts
        font, small, k = L["font"], L["small"], L["k"]
        y_top = int(20 * k)
        x = plot.x
        if sc.inlet is None:
            hdr = [("Looking for the Muse on LSL...  start BlueMuse / OpenMuse", TEXT_DIM)]
        else:
            hdr = []
            if sc.show[0]:
                hdr.append((f"CH1 {sc.ch_label}  1-40 Hz  {UV_STEPS[sc.uv[0]]} uV/div", CH_COLOURS[0]))
            if sc.show[1]:
                hdr.append((f"CH2 alpha {sc.iaf:.1f} Hz (ecHT)  {UV_STEPS[sc.uv[1]]} uV/div",
                            CH_COLOURS[1]))
        for text, col in hdr:
            s = font.render(text, True, col)
            screen.blit(s, (x, y_top))
            x += s.get_width() + int(40 * k)

        run = sc.running and sc.inlet is not None and now - sc.last_data < 3
        badge = font.render(" RUN " if run else (" STOP " if not sc.running else " NO SIGNAL "), True, BG)
        r = badge.get_rect(topright=(plot.right, y_top - int(2 * k)))
        pygame.draw.rect(screen, RUN_GREEN if run else STOP_RED, r.inflate(int(8 * k), int(4 * k)),
                         border_radius=int(4 * k))
        screen.blit(badge, r)
        tb = font.render(f"{MS_STEPS[sc.ms]} ms/div   ROLL   {'SMOOTH' if sc.smooth else 'LIVE'}", True, TEXT_DIM)
        screen.blit(tb, tb.get_rect(topright=(r.left - int(30 * k), y_top)))

        y_bot = plot.bottom + int(20 * k)
        if sc.inlet is not None:
            hold = sc.lag_samples() / sc.fs * 1000
            frame = 1000 / max(clock.get_fps(), 1)
            parts = [f"{sc.name}  {sc.fs:.0f} Hz",
                     f"behind headband: Bluetooth ~30 ms + display {hold + frame:.0f} ms"
                     f" + filter CH1 {sc.filter_ms[0]:.0f} / CH2 {sc.filter_ms[1]:.0f} ms"]
            s = small.render("     ".join(parts), True, TEXT_DIM)
            screen.blit(s, (plot.x, y_bot))
        s = small.render("neurotune", True, TEXT_DIM)
        screen.blit(s, s.get_rect(topright=(plot.right, y_bot)))

        t_draw = time.perf_counter()
        pygame.display.flip()
        if debug:
            stats.append((t_draw - now, time.perf_counter() - t_draw))
            if now - stats_t[0] > 2:
                d = np.array(stats)
                print(f"fps {clock.get_fps():5.1f}  draw {d[:, 0].mean() * 1000:5.1f} ms  "
                      f"flip {d[:, 1].mean() * 1000:5.1f} ms  vsync {vsync['on']}", flush=True)
                stats.clear()
                stats_t[0] = now
                save_still(screen, "scope-debug.png")
        # vsync paces the loop when it's on. Otherwise use the precise busy-wait:
        # Windows' ~15 ms sleep granularity would stack on top and halve the frame rate.
        if vsync["on"]:
            clock.tick()
        else:
            clock.tick_busy_loop(144)
