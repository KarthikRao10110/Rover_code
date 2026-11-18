"""Streaming text-to-speech: Piper -> PipeWire (pw-play).

say() queues sentences. A synth thread renders them while a player thread plays the
previous one. All sentences of one reply go into ONE pw-play stream (with a short natural
pause between them), so there are no clicks or start-up gaps between sentences. The
player publishes a loudness envelope so the face can move its mouth in sync.

synth() + play() let a caller render a reply ahead of time (e.g. while the other robot is
still talking in a debate) and start it later with no synthesis delay.
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
LEAD_S = 0.25        # hand the next sentence to the stream this long before the current one ends
FILL_S = 0.05        # keep at most this much silence buffered ahead while waiting for text

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
        self._proc = None                 # the open pw-play stream (one per reply)
        self._play_end = 0.0              # monotonic time when everything written so far has played
        self._stream_gen = -1             # the _gen the open stream belongs to
        self._pending = 0                 # sentences queued but not finished playing
        self._lock = threading.Lock()
        self._synth_lock = threading.Lock()
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

    def synth(self, text):
        """Render a sentence now (in the caller's thread) without playing it.
        Returns a clip for play(), or None if there's nothing to say."""
        text = clean_for_speech(text)
        if not text:
            return None
        pcm, timeline = self._render(text)
        return (pcm, text, timeline) if pcm else None

    def play(self, clip):
        """Queue a clip from synth(): starts with no synthesis delay."""
        if not clip:
            return
        with self._lock:
            self._pending += 1
            first = self._pending == 1
        if first:
            self.on_state(True)
        pcm, text, timeline = clip
        self._audio_q.put((self._gen, pcm, text, timeline))

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
        self._play_end = 0.0
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
    def _render(self, text):
        """text -> (int16 PCM bytes, viseme timeline). Never raises."""
        try:
            with self._synth_lock:   # espeak-ng (phonemizer) isn't thread-safe
                chunks = list(self.voice.synthesize(text, self.syn, include_alignments=True))
            pcm = b"".join(c.audio_int16_bytes for c in chunks)
            timeline, offset = [], 0
            for c in chunks:
                if c.phoneme_alignments:
                    timeline += [(offset + e, v) for e, v in viseme_timeline(c.phoneme_alignments)]
                offset += len(c.audio_int16_bytes) // 2
            return pcm, timeline
        except Exception as e:  # never let one bad sentence kill the voice
            print(f"[tts] synth failed: {e}")
            return b"", []

    def _synth_loop(self):
        while True:
            gen, text = self._text_q.get()
            if gen != self._gen:
                continue
            pcm, timeline = self._render(text)
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

    @staticmethod
    def _pause_after(text):
        """Breathing gap after a piece: longer after a full sentence than after a clause."""
        t = text.rstrip(" \"')]")
        if t.endswith(("?", "!")):
            return config.TTS_PAUSE_S * 1.2
        if t.endswith("."):
            return config.TTS_PAUSE_S
        return config.TTS_PAUSE_CLAUSE_S

    def _stream(self, gen):
        """The open pw-play stream for this reply (opened on first use)."""
        p = self._proc
        if p is None or self._stream_gen != gen or p.poll() is not None:
            self._close_stream(kill=True)
            p = self._proc = subprocess.Popen(
                ["pw-play", "--raw", "--rate", str(self.rate), "--channels", "1",
                 "--format", "s16", "-"],
                stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self._stream_gen = gen
            self._play_end = time.monotonic()
        return p

    def _write(self, data):
        p = self._proc
        try:
            p.stdin.write(data)
            p.stdin.flush()
        except (BrokenPipeError, OSError, ValueError, AttributeError):
            pass

    def _close_stream(self, kill=False):
        """End the stream; without kill, waits for the buffered audio to finish playing."""
        p, self._proc = self._proc, None
        if not p:
            return
        if kill and p.poll() is None:
            p.kill()
        try:
            p.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()

    def _sleep_until(self, t, gen):
        while gen == self._gen:
            d = t - time.monotonic()
            if d <= 0:
                return
            time.sleep(min(d, 0.02))

    def _play_loop(self):
        silence = bytes(2 * int(FILL_S * self.rate))
        while True:
            try:
                gen, pcm, text, timeline = self._audio_q.get(timeout=0.02)
            except queue.Empty:
                # mid-reply but the next sentence isn't rendered yet: keep the stream fed with
                # a little silence so it never runs dry (no underrun clicks)
                if (self._proc and self._stream_gen == self._gen and self._pending > 0
                        and self._play_end - time.monotonic() < FILL_S):
                    self._write(silence)
                    self._play_end = max(self._play_end, time.monotonic()) + FILL_S
                continue
            if gen != self._gen:
                continue
            if pcm:
                self._stream(gen)
                start = max(time.monotonic(), self._play_end)
                dur = len(pcm) / 2 / self.rate
                pause = self._pause_after(text)
                self.envelope = (start, self._envelope(pcm))
                self._now = (start, dur, text)
                self._visemes = (start, timeline) if timeline else None
                self._write(pcm + bytes(2 * int(pause * self.rate)))
                self._play_end = start + dur + pause
                # hand over to the next sentence just before this one ends, so it joins seamlessly
                self._sleep_until(start + dur - LEAD_S, gen)
                with self._lock:
                    last = self._pending <= 1
                if last:  # nothing else queued: let it finish, then close the stream
                    self._sleep_until(start + dur, gen)
                    if gen == self._gen:
                        self._close_stream()
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
                if self._proc:          # (e.g. an empty last clip) don't leave a stream open
                    self._close_stream()
                self.envelope = None
                self.on_state(False)
