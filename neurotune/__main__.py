"""neurotune - brainwave-locked sound for the Muse S.

  python -m neurotune simulate            fake Muse on LSL (test without the headband)
  python -m neurotune calibrate           2 min eyes closed -> your alpha frequency + best sensor
  python -m neurotune loopback            measure Bluetooth/timestamp delay (optional wiring)
  python -m neurotune run --mode sleep    Elemind-style pink-noise pulses (30 min, fades on sleep)
  python -m neurotune run --mode music    generative music locked to your alpha rhythm
  python -m neurotune analyze FILE.npz    how accurately pulses hit the target phase
  python -m neurotune offline             whole loop on a virtual clock -> WAV + accuracy report
  python -m neurotune ui                  control panel: live session with sound sliders + brain meters
  python -m neurotune scope               HD low-latency oscilloscope of the live EEG
"""
import argparse
import json


def main():
    p = argparse.ArgumentParser(prog="neurotune", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sim = sub.add_parser("simulate", help="stream fake Muse EEG on LSL")
    sim.add_argument("--iaf", type=float, default=10.0)

    cal = sub.add_parser("calibrate", help="find your alpha frequency")
    cal.add_argument("--seconds", type=float, default=120)

    lb = sub.add_parser("loopback", help="measure EEG timestamp delay")
    lb.add_argument("--clicks", type=int, default=40)
    lb.add_argument("--device", default=None)

    r = sub.add_parser("run", help="live session")
    r.add_argument("--mode", choices=["sleep", "music"], default="sleep")
    r.add_argument("--minutes", type=float, default=30)
    r.add_argument("--volume", type=float, default=-12, help="dB, default -12")
    r.add_argument("--target", default="trough",
                   help="where the evoked response lands: trough (lower alpha, sleep), peak, or an onset phase in degrees")
    r.add_argument("--gate", type=float, default=None, help="alpha strength gate (0 = pulse every cycle)")
    r.add_argument("--device", default=None, help="sound output device name or index")
    r.add_argument("--osc", default=None, help="also send pulses/state as OSC, e.g. 127.0.0.1:9000")

    a = sub.add_parser("analyze", help="phase accuracy of a saved session")
    a.add_argument("file")

    o = sub.add_parser("offline", help="simulated closed loop -> WAV + report")
    o.add_argument("--mode", choices=["sleep", "music"], default="sleep")
    o.add_argument("--minutes", type=float, default=3)
    o.add_argument("--out", default="sessions/offline")

    sub.add_parser("devices", help="list sound devices")
    sub.add_parser("ui", help="control panel for a live session")

    sc = sub.add_parser("scope", help="HD low-latency oscilloscope of the live EEG")
    sc.add_argument("--channel", default=None, help="TP9, AF7, AF8 or TP10 (default: from config)")
    sc.add_argument("--mains", type=float, default=50.0, help="mains frequency to notch out (50 UK, 60 US)")

    args = p.parse_args()
    if args.cmd == "simulate":
        from .sim import stream_lsl
        stream_lsl(args.iaf)
    elif args.cmd == "calibrate":
        from .realtime import calibrate
        calibrate(args.seconds)
    elif args.cmd == "loopback":
        from .realtime import loopback
        loopback(args.clicks, args.device)
    elif args.cmd == "run":
        from .realtime import run
        run(args.mode, args.minutes, args.volume, args.target, args.device, gate=args.gate, osc=args.osc)
    elif args.cmd == "analyze":
        from .analyze import analyze
        print(json.dumps(analyze(args.file), indent=2, default=float))
    elif args.cmd == "offline":
        from .sim import run_offline
        from .analyze import analyze
        s = run_offline(args.out, args.minutes, args.mode)
        print(json.dumps(s, indent=2))
        print(json.dumps(analyze(f"{args.out}/session_{args.mode}.npz"), indent=2, default=float))
    elif args.cmd == "ui":
        from .ui import main as ui_main
        ui_main()
    elif args.cmd == "scope":
        from .oscilloscope import main as scope_main
        scope_main(args.channel, args.mains)
    elif args.cmd == "devices":
        import sounddevice as sd
        print(sd.query_devices())


if __name__ == "__main__":
    main()
