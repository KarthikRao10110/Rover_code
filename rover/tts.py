"""Streaming text-to-speech: Piper -> PipeWire (pw-play).

say() queues sentences. A synth thread renders them while a player thread plays the
previous one, so there are no gaps between sentences. The player publishes a loudness
envelope so the face can move its mouth in sync.
"""

import queue
import re
import subprocess
import threading
import time

import numpy as np
import onnxruntime as ort
from piper import PiperVoice, SynthesisConfig

from . import config

ENV_FPS = 30
BT_LATENCY_S = 0.15  # Bluetooth delay; shifts the mouth so it lines up with the sound

# espeak IPA phoneme -> viseme (mouth shape). Stress/length marks take the next phoneme's shape.
VISEMES = {}
for v, chars in {
    "MBP": "mbp", "FV": "fv", "TH": "θð", "L": "lɫ",
    "O": "oɔɒ", "U": "uʊwɥ", "AA": "aɑæʌɐ", "E": "eɛəɜɚɝ", "I": "iɪjy",
    "C": "tdnszkgŋhrɹɾʃʒxçʔ",
}.items():
    for ch in chars:
        VISEMES[ch] = v
MARKS = set("ˈˌːˑ")


def viseme_timeline(alignments):
    """[(end_sample, viseme), ...] from Piper phoneme alignments."""
    out, t = [], 0
    pending = 0  # samples of stress marks waiting for the next real phoneme
    for a in alignments:
        n = int(a.num_samples)
        if a.phoneme in MARKS:
            pending += n
            continue
        v = VISEMES.get(a.phoneme, "rest")
        if pending and v != "rest":
            n += pending
        elif pending and out:
            out.append((t + pending, out[-1][1]))
            t += pending
        pending = 0
        t += n
        out.append((t, v))
    return out


_EMOJI = re.compile("[\U0001F000-\U0001FFFF☀-➿️]")


def clean_for_speech(text):
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"[*_#`>|~]", "", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)  # markdown links
    text = _EMOJI.sub("", text)
    text = text.replace("&", " and ")
    return re.sub(r"\s+", " ", text).strip()


class Speaker:
    def __init__(self, on_state=None):
        self.voice = PiperVoice.load(config.TTS_VOICE, include_alignments=True)
        # Piper defaults to every CPU core, which starves the LLM generating the next sentence
        # at the same time. 1 thread still synthesizes ~4x faster than real time on a Pi 5.
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.voice.session = ort.InferenceSession(config.TTS_VOICE, sess_options=so,
                                                  providers=["CPUExecutionProvider"])
        self.syn = SynthesisConfig(length_scale=config.TTS_SPEED, noise_scale=config.TTS_EXPRESSION,
                                   noise_w_scale=config.TTS_RHYTHM)  # livelier, less "announcement"
        self.rate = self.voice.config.sample_rate
        self.on_state = on_state or (lambda speaking: None)
        self._text_q = queue.Queue()
        self._audio_q = queue.Queue(maxsize=3)
        self._gen = 0                     # bumps on cancel(); stale work is dropped
        self._proc = None
        self._pending = 0                 # sentences queued but not finished playing
        self._lock = threading.Lock()
        self.envelope = None              # (start_time, np.array of 0..1 levels at ENV_FPS)
        self._said = []                   # sentences fully spoken this turn (for subtitles)
        self._now = None                  # (start_time, duration_s, text) of the sentence playing
        self._visemes = None              # (start_time, [(end_sample, viseme)]) for lip sync
        self._recent = []                 # [(end_time, text)] of played sentences (echo filter)
        self.last_spoke = 0.0
        threading.Thread(target=self._synth_loop, name="tts-synth", daemon=True).start()
        threading.Thread(target=self._play_loop, name="tts-play", daemon=True).start()

    # ---- public ----------------------------------------------------------------
    def say(self, text):
        text = clean_for_speech(text)
        if not text:
            return
        with self._lock:
            self._pending += 1
            first = self._pending == 1
        if first:
            self.on_state(True)
        self._text_q.put((self._gen, text))

    def new_turn(self):
        """Clear the subtitles (call when a new reply starts)."""
        self._said = []
        self._now = None

    def caption(self):
        """Text spoken so far, revealed word by word in step with the audio."""
        text = " ".join(self._said)
        cur = self._now
        if cur:
            start, dur, sentence = cur
            words = sentence.split()
            frac = (time.monotonic() - start - BT_LATENCY_S) / max(dur, 0.1)
            n = max(0, min(len(words), int(frac * len(words) + 0.999)))
            if n:
                text = (text + " " + " ".join(words[:n])).strip()
        return text

    def recent_sentences(self, seconds=15):
        now = time.monotonic()
        out = [t for end, t in self._recent if now - end < seconds]
        if self._now:
            out.append(self._now[2])
        return out

    def recent_text(self, seconds=15):
        """What the robot said in the last few seconds, including the sentence playing now."""
        now = time.monotonic()
        parts = [t for end, t in self._recent if now - end < seconds]
        if self._now:
            parts.append(self._now[2])
        return " ".join(parts)

    def speaking(self):
        return self._pending > 0

    def wait(self, timeout=60):
        end = time.monotonic() + timeout
        while self.speaking() and time.monotonic() < end:
            time.sleep(0.03)

    def cancel(self):
        """Stop talking immediately and drop anything queued."""
        self._gen += 1
        for q in (self._text_q, self._audio_q):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        p = self._proc
        if p and p.poll() is None:
            p.kill()
        with self._lock:
            self._pending = 0
        if self._now:  # remember the cut-off sentence for the echo filter
            self._recent.append((time.monotonic(), self._now[2]))
            self.last_spoke = time.monotonic()
        self.envelope = None
        self._now = None
        self._visemes = None
        self.on_state(False)

    def viseme(self):
        """Current mouth shape for lip sync ("rest" when silent)."""
        vis = self._visemes
        if not vis:
            return "rest"
        start, timeline = vis
        sample = (time.monotonic() - start - BT_LATENCY_S) * self.rate
        if sample < 0:
            return "rest"
        for end, v in timeline:
            if sample < end:
                return v
        return "rest"

    def mouth_level(self):
        env = self.envelope
        if env is None:
            return 0.0
        start, levels = env
        i = int((time.monotonic() - start - BT_LATENCY_S) * ENV_FPS)
        return float(levels[i]) if 0 <= i < len(levels) else 0.0

    # ---- threads -----------------------------------------------------------------
    def _synth_loop(self):
        while True:
            gen, text = self._text_q.get()
            if gen != self._gen:
                continue
            try:
                chunks = list(self.voice.synthesize(text, self.syn, include_alignments=True))
                pcm = b"".join(c.audio_int16_bytes for c in chunks)
                timeline, offset = [], 0
                for c in chunks:
                    if c.phoneme_alignments:
                        timeline += [(offset + e, v) for e, v in viseme_timeline(c.phoneme_alignments)]
                    offset += len(c.audio_int16_bytes) // 2
            except Exception as e:  # never let one bad sentence kill the voice
                print(f"[tts] synth failed: {e}")
                pcm, timeline = b"", []
            if gen == self._gen:
                self._audio_q.put((gen, pcm, text, timeline))

    def _envelope(self, pcm):
        a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        hop = self.rate // ENV_FPS
        n = len(a) // hop
        if n == 0:
            return np.zeros(1)
        rms = np.sqrt(np.mean(a[: n * hop].reshape(n, hop) ** 2, axis=1))
        return np.clip(rms / (rms.max() + 1e-6), 0, 1)

    def _play_loop(self):
        while True:
            gen, pcm, text, timeline = self._audio_q.get()
            if gen == self._gen and pcm:
                t0 = time.monotonic()
                self.envelope = (t0, self._envelope(pcm))
                self._now = (t0, len(pcm) / 2 / self.rate, text)
                self._visemes = (t0, timeline) if timeline else None
                self._proc = subprocess.Popen(
                    ["pw-play", "--raw", "--rate", str(self.rate), "--channels", "1",
                     "--format", "s16", "-"],
                    stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
                try:
                    self._proc.stdin.write(pcm)
                    self._proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
                self._proc.wait()
            if gen != self._gen:
                continue
            if pcm:
                self._said.append(text)
                self._now = None
                self._visemes = None
                self.last_spoke = time.monotonic()
                self._recent = [r for r in self._recent if self.last_spoke - r[0] < 20] + [(self.last_spoke, text)]
            with self._lock:
                self._pending = max(0, self._pending - 1)
                done = self._pending == 0
            if done:
                self.envelope = None
                self.on_state(False)
