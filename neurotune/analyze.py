"""After a session: how accurately did pulses land on the intended phase?

Uses a zero-phase (non-causal) filter + Hilbert transform on the recorded
EEG as the reference, which is the standard post-hoc check (Elemind's trial
reported 0.5 +/- 6.9 deg onset error on its own dedicated hardware).
For simulated sessions it also scores against the ground-truth oscillation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .dsp import TWO_PI, offline_phase, wrap, wrap_pm


def _circ(err):
    z = np.mean(np.exp(1j * err))
    mean = np.angle(z)
    sd = np.sqrt(-2 * np.log(np.abs(z) + 1e-12))
    return np.rad2deg(mean), np.rad2deg(sd), np.abs(z)


def target_onset(target, p1, freq):
    if str(target) in ("trough", "peak"):
        land = np.pi if str(target) == "trough" else 0.0
        return wrap(land - TWO_PI * freq * p1)
    return np.deg2rad(float(target)) * np.ones_like(freq)


def analyze(npz_path: str, eeg_latency_s: float | None = None, plot: bool = True) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    fs, iaf = float(d["fs"]), float(d["iaf"])
    eeg, pulses = d["eeg"], d["pulses"]
    chans = list(d["channels"])
    ch = chans.index(str(d["channel"])) + 1
    if eeg_latency_s is None:
        eeg_latency_s = float(d["eeg_latency"]) if "eeg_latency" in d else float(d.get("ble_delay", 0.0))
    played = pulses[pulses[:, 1] == 1]
    t_on, freq = played[:, 0], played[:, 2]
    tgt = target_onset(d["target"], float(d["p1"]), freq)

    res = {"pulses": int(len(played))}
    # Reference from recorded EEG (what you can do with real data).
    t_eeg = eeg[:, 0] - eeg_latency_s
    ph = offline_phase(eeg[:, ch], fs, (iaf - 2, iaf + 2))
    keep = (t_on > t_eeg[0] + 2) & (t_on < t_eeg[-1] - 2)
    idx = np.searchsorted(t_eeg, t_on[keep])
    err = wrap_pm(ph[idx] - tgt[keep])
    m, sd, r = _circ(err)
    res["eeg_ref"] = {"mean_err_deg": round(m, 1), "circ_sd_deg": round(sd, 1), "locking_R": round(r, 3)}
    res["suggested_phase_bias_deg"] = round(-m, 1)

    if "truth" in d:   # simulator ground truth
        from scipy.signal import hilbert
        tph = np.angle(hilbert(d["truth"]))
        k = np.clip(np.round(t_on[keep] * fs).astype(int), 0, len(tph) - 1)
        e2 = wrap_pm(tph[k] - tgt[keep])
        m2, sd2, r2 = _circ(e2)
        res["ground_truth"] = {"mean_err_deg": round(m2, 1), "circ_sd_deg": round(sd2, 1), "locking_R": round(r2, 3)}

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(projection="polar")
        ax.hist(np.deg2rad(np.rad2deg(err)), bins=36, color="#3b6fb6", alpha=0.85)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(f"Pulse phase error vs target\nmean {m:+.0f} deg, spread {sd:.0f} deg, n={len(err)}")
        out = Path(npz_path).with_suffix(".phase.png")
        fig.tight_layout()
        fig.savefig(out, dpi=130)
        res["plot"] = str(out)
    return res


def analyze_loopback(npz_path: str) -> dict:
    """Find the delay between logged click times and clicks seen in the EEG.

    The click is fed electrically into an electrode (see README), so the lag
    is the total timestamp error of the chain: set it as --eeg-latency.
    """
    from scipy import signal
    d = np.load(npz_path, allow_pickle=True)
    eeg, clicks, fs = d["eeg"], d["clicks"], float(d["fs"])
    sos = signal.butter(4, 40, "highpass", fs=fs, output="sos")
    energy = np.abs(signal.sosfiltfilt(sos, eeg[:, 1:], axis=0)).sum(1)
    lags = []
    for c in clicks:
        sel = (eeg[:, 0] > c - 0.1) & (eeg[:, 0] < c + 0.4)
        if sel.sum() < 20:
            continue
        seg, ts = energy[sel], eeg[sel, 0]
        base = np.median(seg[: int(0.08 * fs)])
        hit = np.argmax(seg > base + 6 * np.median(np.abs(seg - base)) + 1e-9)
        if hit > 0:
            lags.append(ts[hit] - c)
    lags = np.array(lags)
    return {"clicks": int(len(clicks)), "detected": int(len(lags)),
            "eeg_latency_ms": round(float(np.median(lags) * 1000), 1) if len(lags) else None,
            "spread_ms": round(float(np.std(lags) * 1000), 1) if len(lags) else None}
