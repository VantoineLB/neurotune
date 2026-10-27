"""Signal processing: real-time phase tracking and band power.

The phase tracker is the endpoint-corrected Hilbert transform (ecHT) from
Schreglmann et al. 2021 (Nature Communications), the method Elemind's
published trials use. It estimates the instantaneous phase of a narrowband
oscillation at the *last* sample of a window, which a normal Hilbert
transform gets badly wrong because of edge (Gibbs) effects.

Phase convention used throughout: 0 rad = oscillation peak, pi = trough.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

TWO_PI = 2 * np.pi


class ECHT:
    """Endpoint-corrected Hilbert transform for one window length."""

    def __init__(self, fs: float, n: int, band: tuple[float, float], order: int = 2):
        self.fs = fs
        self.n = n
        self.set_band(band, order)

    def set_band(self, band: tuple[float, float], order: int = 2):
        self.band = band
        b, a = signal.butter(order, band, btype="bandpass", fs=self.fs)
        # Causal filter's frequency response, sampled on the FFT grid.
        _, h = signal.freqz(b, a, worN=self.n, whole=True)
        # Analytic-signal mask (zero negative frequencies, double positive).
        mask = np.zeros(self.n)
        mask[0] = 1.0
        if self.n % 2 == 0:
            mask[1 : self.n // 2] = 2.0
            mask[self.n // 2] = 1.0
        else:
            mask[1 : (self.n + 1) // 2] = 2.0
        self._kernel = mask * h

    def last(self, x: np.ndarray) -> tuple[float, float]:
        """Return (phase, envelope) at the final sample of x (len == n)."""
        x = np.asarray(x, dtype=float)
        x = x - x.mean()
        y = np.fft.ifft(np.fft.fft(x) * self._kernel)
        z = y[-1]
        return float(np.angle(z)), float(np.abs(z))


def wrap(phase):
    """Wrap angle(s) to [0, 2pi)."""
    return np.mod(phase, TWO_PI)


def wrap_pm(phase):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(phase) + np.pi) % TWO_PI - np.pi


def band_powers(x: np.ndarray, fs: float) -> dict[str, float]:
    """Absolute power in classic EEG bands (Welch)."""
    nper = min(len(x), int(fs * 2))
    f, p = signal.welch(x - np.mean(x), fs=fs, nperseg=nper)

    def bp(lo, hi):
        sel = (f >= lo) & (f < hi)
        return float(np.trapezoid(p[sel], f[sel])) if sel.any() else 0.0

    return {
        "delta": bp(1, 4),
        "theta": bp(4, 7.5),
        "alpha": bp(7.5, 12.5),
        "beta": bp(13, 30),
    }


def find_iaf(x: np.ndarray, fs: float, lo=7.5, hi=12.5) -> tuple[float, float]:
    """Individual alpha frequency: spectral peak in [lo, hi] Hz.

    Returns (iaf_hz, peak_snr) where SNR compares the peak with the
    1/f background fitted on 3-30 Hz excluding the alpha band.
    """
    f, p = signal.welch(x - np.mean(x), fs=fs, nperseg=int(fs * 4))
    sel = (f >= lo) & (f <= hi)
    iaf = float(f[sel][np.argmax(p[sel])])
    fit = (f >= 3) & (f <= 30) & ~sel
    coef = np.polyfit(np.log(f[fit]), np.log(p[fit] + 1e-20), 1)
    background = np.exp(np.polyval(coef, np.log(iaf)))
    snr = float(p[sel].max() / background)
    return iaf, snr


def offline_phase(x: np.ndarray, fs: float, band: tuple[float, float]) -> np.ndarray:
    """Zero-phase (non-causal) reference phase for accuracy analysis."""
    sos = signal.butter(4, band, btype="bandpass", fs=fs, output="sos")
    return np.angle(signal.hilbert(signal.sosfiltfilt(sos, x - np.mean(x))))
