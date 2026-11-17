"""Record a few seconds from the mic, show the level, transcribe it with Parakeet.

    .venv/bin/python -m tools.mic_test               # default mic, 5 s
    .venv/bin/python -m tools.mic_test --secs 8 --target <pipewire source name>
"""

import argparse
import subprocess
import time

import numpy as np

from rover.stt import make_recognizer as parakeet

RATE = 16000


def record(secs, target=None):
    cmd = ["pw-record", "--rate", str(RATE), "--channels", "1", "--format", "s16", "--raw"]
    if target:
        cmd += ["--target", target]
    cmd.append("-")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    need = int(secs * RATE) * 2
    buf = b""
    while len(buf) < need:
        chunk = p.stdout.read(need - len(buf))
        if not chunk:
            break
        buf += chunk
    p.terminate()
    return np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=5)
    ap.add_argument("--target", help="PipeWire source (see: wpctl status)")
    args = ap.parse_args()

    rec = parakeet()
    print(f"Get ready to speak for {args.secs:.0f} seconds...", flush=True)
    for n in (3, 2, 1):
        print(f"  {n}", flush=True)
        time.sleep(1)
    print("RECORDING... speak now", flush=True)
    audio = record(args.secs, args.target)
    peak, rms = float(np.abs(audio).max()), float(np.sqrt(np.mean(audio**2)))
    print(f"got {len(audio) / RATE:.1f}s  peak={peak:.2f}  rms={rms:.3f}"
          + ("   <-- very quiet, check mic / gain" if peak < 0.1 else ""))
    t = time.perf_counter()
    s = rec.create_stream()
    s.accept_waveform(RATE, audio)
    rec.decode_stream(s)
    print(f"transcribed in {time.perf_counter() - t:.2f}s:\n  \"{s.result.text.strip()}\"")


if __name__ == "__main__":
    main()
