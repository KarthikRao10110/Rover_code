"""Measure the speech-recognition accuracy of a microphone.

    .venv/bin/python -m tools.stt_check                 # current/default mic
    .venv/bin/python -m tools.stt_check --target <pipewire source>   # e.g. a USB mic

Accuracy: read PASSAGE aloud at normal speed and distance. Prints word error rate (WER)
and exactly which words were missed or misheard. The recording is saved in logs/ so the
same audio can be compared across settings.

"""

import argparse
import os
import re
import time

import numpy as np
import soundfile as sf

from rover.audio import ensure_bt, mic_target
from rover.stt import RATE, make_recognizer, mic_chunks

PASSAGE = ("Renewable energy like solar and wind is getting cheaper every year, "
           "but many countries still depend on coal for most of their electricity. "
           "Switching completely will take decades and a lot of new infrastructure.")
def norm(t):
    return re.sub(r"[^a-z0-9' ]", " ", t.lower().replace("-", " ")).split()


def word_diff(ref, hyp):
    """Levenshtein alignment over words -> (wer, list of (op, ref_word, hyp_word))."""
    r, h = norm(ref), norm(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = range(len(r) + 1)
    d[0, :] = range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    ops, i, j = [], len(r), len(h)
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            ops.append(("ok" if r[i - 1] == h[j - 1] else "sub", r[i - 1], h[j - 1])); i -= 1; j -= 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            ops.append(("missed", r[i - 1], "")); i -= 1
        else:
            ops.append(("extra", "", h[j - 1])); j -= 1
    ops.reverse()
    return d[len(r), len(h)] / max(1, len(r)), ops


def record(target, secs):
    out = []
    end = time.monotonic() + secs
    for c in mic_chunks(target):
        out.append(c)
        if time.monotonic() > end:
            break
    return np.concatenate(out)


def countdown(msg, n=3):
    print(msg, flush=True)
    for k in range(n, 0, -1):
        print(f"  {k}", flush=True)
        time.sleep(1)


def transcribe(rec, audio):
    s = rec.create_stream()
    s.accept_waveform(RATE, audio)
    rec.decode_stream(s)
    return s.result.text.strip()


def test_accuracy(rec, target, secs):
    print("\n=== ACCURACY TEST ===")
    print(f'Read this aloud at a normal pace:\n\n  "{PASSAGE}"\n')
    countdown("Starting in", 3)
    print(f"RECORDING ({secs:.0f} s)... read now", flush=True)
    audio = record(target, secs)
    os.makedirs("logs", exist_ok=True)
    path = time.strftime("logs/stt_check_%H%M%S.wav")
    sf.write(path, audio, RATE)
    peak, rms = float(np.abs(audio).max()), float(np.sqrt(np.mean(audio**2)))
    t = time.perf_counter()
    text = transcribe(rec, audio)
    wer, ops = word_diff(PASSAGE, text)
    print(f"\nlevel: peak {peak:.2f}, rms {rms:.3f}" + ("   <- quiet, speak up or move closer" if peak < 0.1 else ""))
    print(f"heard ({time.perf_counter() - t:.1f}s): {text}")
    print(f"\nWORD ACCURACY: {100 * (1 - wer):.0f}%   (WER {100 * wer:.0f}%)")
    errs = [o for o in ops if o[0] != "ok"]
    for op, r, h in errs:
        print(f"  {op:6}  said '{r}'" + (f" -> heard '{h}'" if h else "") if op != "extra" else f"  extra   heard '{h}'")
    print(f"recording saved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", help="PipeWire source name (default: configured mic)")
    ap.add_argument("--secs", type=float, default=15)
    args = ap.parse_args()
    target = args.target or (mic_target() if ensure_bt() else None)
    print(f"mic: {target or 'system default'}")
    rec = make_recognizer()
    test_accuracy(rec, target, args.secs)


if __name__ == "__main__":
    main()
