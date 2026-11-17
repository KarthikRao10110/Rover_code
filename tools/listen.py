"""Live transcription: speak, pause, see the text. Ctrl+C to quit.

    .venv/bin/python -m tools.listen
    .venv/bin/python -m tools.listen --file some.wav   # run the same pipeline on a file
"""

import argparse
import time

import numpy as np
import soundfile as sf

from rover.stt import RATE, WINDOW, Listener, ListenerConfig


def file_chunks(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != RATE:  # simple linear resample, fine for testing
        n = int(len(audio) * RATE / sr)
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
    audio = np.concatenate([audio, np.zeros(RATE * 2, dtype=np.float32)])  # trailing silence closes the segment
    for i in range(0, len(audio) - WINDOW + 1, WINDOW):
        yield audio[i:i + WINDOW]


def timed(chunks, secs):
    end = time.monotonic() + secs
    for c in chunks:
        if time.monotonic() > end:
            return
        yield c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--target", help="PipeWire mic source name")
    ap.add_argument("--duration", type=float, help="stop after this many seconds")
    ap.add_argument("--silence", type=float, default=0.7, help="pause (s) that ends a phrase")
    args = ap.parse_args()

    t = time.perf_counter()
    lst = Listener(ListenerConfig(mic_target=args.target, min_silence=args.silence))
    print(f"ready (models loaded in {time.perf_counter() - t:.1f}s) — speak; Ctrl+C to quit", flush=True)
    from rover.stt import mic_chunks
    src = file_chunks(args.file) if args.file else mic_chunks(args.target)
    if args.duration:
        src = timed(src, args.duration)
    try:
        for text in lst.utterances(src):
            s = lst.last_stats
            print(f"[{s['audio_s']:.1f}s audio, stt {s['stt_s']:.2f}s] {text}", flush=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
