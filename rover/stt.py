"""Always-on listening: mic -> Silero VAD -> Parakeet -> text callback.

    from rover.stt import Listener
    for text in Listener().utterances():
        print(text)
"""

import os
import queue
import select
import subprocess
import threading
import time
from dataclasses import dataclass

import numpy as np
import sherpa_onnx

RATE = 16000
WINDOW = 512  # Silero VAD frame size at 16 kHz (32 ms)
PREROLL_S = 0.4   # VAD fires ~0.25-0.3 s into speech: add this much audio before each phrase
POSTROLL_S = 0.2  # ...and a little after, so trailing soft sounds aren't clipped
RING_S = 70       # seconds of recent audio kept for that
MODELS = "models"


@dataclass
class ListenerConfig:
    mic_target: str | None = None    # PipeWire source; None = system default mic
    vad_threshold: float = 0.5       # speech probability threshold (raise in noisy rooms)
    min_silence: float = 0.9         # seconds of silence that ends a phrase (agent joins unfinished ones)
    min_speech: float = 0.25         # ignore blips shorter than this
    max_speech: float = 45.0         # force a cut on very long monologues
    min_rms: float = 0.004           # drop near-silent segments (VAD false triggers)
    threads: int = 4
    silent_timeout: float = 3.0      # no data at all from the mic for this long -> error
    save_dir: str | None = None      # keep the last few phrases as WAV files (diagnostics)
    save_keep: int = 30


def make_recognizer(threads=4):
    d = f"{MODELS}/parakeet-tdt-0.6b-v2"
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=f"{d}/encoder.int8.onnx", decoder=f"{d}/decoder.int8.onnx",
        joiner=f"{d}/joiner.int8.onnx", tokens=f"{d}/tokens.txt",
        num_threads=threads, model_type="nemo_transducer")


def make_vad(cfg: ListenerConfig):
    c = sherpa_onnx.VadModelConfig()
    c.silero_vad.model = f"{MODELS}/silero_vad.onnx"
    c.silero_vad.threshold = cfg.vad_threshold
    c.silero_vad.min_silence_duration = cfg.min_silence
    c.silero_vad.min_speech_duration = cfg.min_speech
    c.silero_vad.max_speech_duration = cfg.max_speech
    c.silero_vad.window_size = WINDOW
    c.sample_rate = RATE
    c.num_threads = 1
    return sherpa_onnx.VoiceActivityDetector(c, buffer_size_in_seconds=60)


def mic_chunks(target=None, silent_timeout=3.0, paused=None, grace=4.0):
    """Yield float32 mono 16 kHz chunks of WINDOW samples from PipeWire.

    While paused() is True the recording is CLOSED (so a Bluetooth speaker can stay in music
    mode) and None is yielded every ~50 ms; it reopens when paused() turns False.
    Raises RuntimeError if the mic delivers no data for silent_timeout seconds
    (e.g. Bluetooth dropped: PipeWire then falls back to a source that never sends audio).
    Right after (re)opening, `grace` extra seconds are allowed (Bluetooth profile switch)."""
    cmd = ["pw-record", "--rate", str(RATE), "--channels", "1",
           "--format", "s16", "--raw", "--latency", "32ms"]
    if target:
        cmd += ["--target", target]
    cmd.append("-")
    nbytes = WINDOW * 2
    p = None

    def close():
        nonlocal p
        if p:
            p.terminate()
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                p.kill()
            p = None

    try:
        while True:
            if paused and paused():
                close()
                time.sleep(0.05)
                yield None
                continue
            if p is None:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                deadline = time.monotonic() + silent_timeout + grace
            fd = p.stdout.fileno()
            buf = b""
            while len(buf) < nbytes:
                if paused and paused():
                    break
                ready, _, _ = select.select([fd], [], [], 0.1)
                if not ready:
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"no audio from microphone for {silent_timeout:.0f} s")
                    continue
                chunk = os.read(fd, nbytes - len(buf))
                if not chunk:
                    raise RuntimeError("microphone stream ended (device disconnected?)")
                buf += chunk
                deadline = time.monotonic() + silent_timeout
            if len(buf) < nbytes:
                continue                          # paused mid-read: drop the partial chunk
            yield np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
    finally:
        close()


class Listener:
    """on_partial(text): optional callback with the live transcript of the phrase still being spoken."""

    def __init__(self, cfg: ListenerConfig | None = None, recognizer=None, partial_every=0.8):
        self.cfg = cfg or ListenerConfig()
        self.recognizer = recognizer or make_recognizer(self.cfg.threads)
        self.vad = make_vad(self.cfg)
        self.muted = False          # half-duplex fallback: drop audio while True
        self.close_when_muted = False  # True: close the recording while muted (BT music mode)
        self.last_stats = {}
        self.level = 0.0            # live mic loudness (RMS) for the face
        self.partial_every = partial_every
        self.on_partial = None
        self.partials_enabled = lambda: True   # e.g. off while the LLM is busy (CPU)
        self.max_phrase_s = lambda: None       # skip (don't transcribe) phrases longer than this
        self._asr_lock = threading.Lock()
        self._partial_q = queue.Queue(maxsize=1)
        threading.Thread(target=self._partial_worker, name="stt-partial", daemon=True).start()

    def transcribe(self, samples):
        with self._asr_lock:
            s = self.recognizer.create_stream()
            s.accept_waveform(RATE, samples)
            self.recognizer.decode_stream(s)
            return s.result.text.strip()

    def _partial_worker(self):
        while True:
            gen, samples = self._partial_q.get()
            text = self.transcribe(samples)
            if gen == self._phrase_gen and self.on_partial and text:
                self.on_partial(text)

    def utterances(self, chunks=None):
        """Yield one transcript per spoken phrase. `chunks` defaults to the live mic."""
        chunks = chunks if chunks is not None else mic_chunks(
            self.cfg.mic_target, self.cfg.silent_timeout,
            paused=lambda: self.muted and self.close_when_muted)
        self._phrase_gen = 0
        preroll, live, in_speech, since_partial = [], [], False, 0.0
        ring = np.zeros(RING_S * RATE, dtype=np.float32)   # circular buffer of fed audio
        fed = 0                                             # samples fed to the VAD so far
        dt = WINDOW / RATE
        for chunk in chunks:
            if chunk is None:                   # recording closed while muted
                self.level = 0.0
                self.vad.reset()
                fed = 0
                continue
            self.level = float(np.sqrt(np.mean(chunk**2)))
            if self.muted:
                self.vad.reset()
                fed = 0
                continue
            self.vad.accept_waveform(chunk)
            pos = fed % len(ring)
            ring[pos:pos + len(chunk)] = chunk[: len(ring) - pos]
            if pos + len(chunk) > len(ring):
                ring[: pos + len(chunk) - len(ring)] = chunk[len(ring) - pos:]
            fed += len(chunk)

            # rolling partial transcripts while someone is talking
            if self.vad.is_speech_detected():
                if not in_speech:
                    in_speech, live, since_partial = True, list(preroll), 0.0
                live.append(chunk)
                since_partial += dt
                if self.on_partial and since_partial >= self.partial_every and self.partials_enabled():
                    since_partial = 0.0
                    try:
                        self._partial_q.put_nowait((self._phrase_gen, np.concatenate(live)))
                    except queue.Full:
                        pass  # previous partial still decoding; skip this one
            elif in_speech:
                in_speech = False
            preroll = (preroll + [chunk])[-10:]

            while not self.vad.empty():
                self._phrase_gen += 1  # stale partials for this phrase are dropped
                front = self.vad.front
                seg = np.array(front.samples, dtype=np.float32)
                start = int(front.start)
                self.vad.pop()
                # widen the phrase with audio from just before/after what the VAD kept
                a = max(0, start - int(PREROLL_S * RATE), fed - len(ring) + 1)
                b = min(fed, start + len(seg) + int(POSTROLL_S * RATE))
                if a < start and b > a:
                    idx = np.arange(a, b) % len(ring)
                    seg = ring[idx].copy()
                rms = float(np.sqrt(np.mean(seg**2)))
                if rms < self.cfg.min_rms:
                    continue
                limit = self.max_phrase_s()
                if limit and len(seg) / RATE > limit:
                    continue
                t = time.perf_counter()
                text = self.transcribe(seg)
                self.last_stats = {"audio_s": len(seg) / RATE, "stt_s": time.perf_counter() - t, "rms": rms}
                if self.cfg.save_dir:
                    self._save(seg, text)
                if text:
                    yield text

    def _save(self, seg, text):
        try:
            import soundfile as sf
            os.makedirs(self.cfg.save_dir, exist_ok=True)
            name = time.strftime("%Y%m%d_%H%M%S")
            sf.write(os.path.join(self.cfg.save_dir, name + ".wav"), seg, RATE)
            with open(os.path.join(self.cfg.save_dir, name + ".txt"), "w") as f:
                f.write(text + "\n")
            files = sorted(f for f in os.listdir(self.cfg.save_dir) if f.endswith(".wav"))
            for old in files[: -self.cfg.save_keep]:
                for ext in (".wav", ".txt"):
                    try:
                        os.remove(os.path.join(self.cfg.save_dir, old[:-4] + ext))
                    except OSError:
                        pass
        except Exception as e:
            print(f"[stt] could not save phrase: {e}")

    @property
    def hearing_speech(self):
        return self.vad.is_speech_detected()
