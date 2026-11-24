"""Hardware configuration. All pin numbers are BCM GPIO numbers (not physical pins)."""

# --- L298N motor driver -------------------------------------------------------
# Left side: two left motors in parallel on OUT1/OUT2
LEFT_IN1 = 17      # physical pin 11
LEFT_IN2 = 27      # physical pin 13
LEFT_EN = 12       # physical pin 32 (ENA, PWM)

# Right side: two right motors in parallel on OUT3/OUT4
RIGHT_IN3 = 22     # physical pin 15
RIGHT_IN4 = 23     # physical pin 16
RIGHT_EN = 13      # physical pin 33 (ENB, PWM)

# Flip these if a side drives backwards when commanded forward
# (run tools/motor_test.py to find out).
LEFT_INVERTED = True   # left motors are wired opposite to the right ones
RIGHT_INVERTED = False

# PWM frequency on ENA/ENB in Hz. L298N is happy around 500-2000 Hz.
PWM_FREQUENCY = 1000

# Below this duty cycle the motors usually just buzz without turning.
# Any non-zero command is mapped into [MIN_DUTY, 1.0]. Tune with motor_test.py.
MIN_DUTY = 0.35

# Max change in motor command per second (1.0 = full speed).
# 4.0 means 0 -> full in 0.25 s. Emergency stop bypasses this.
ACCEL_PER_SEC = 4.0

# --- Ultrasonic sensors (HC-SR04, echo through 1k/2k divider) ------------------
# (trigger, echo). Names are free; the direction each one faces is in ULTRASONIC_ANGLES.
ULTRASONIC = {
    "front_left": (5, 6),
    "front_center": (19, 26),
    "front_right": (16, 20),
    "rear": (24, 25),
}

# --- Teleop -----------------------------------------------------------------
DEFAULT_SPEED = 0.6        # speed limit for normal driving (0..1)
BOOST_SPEED = 1.0          # speed while boost is held
TURN_GAIN_MOVING = 0.5     # how hard to turn while driving forward/back
TURN_GAIN_SPIN = 0.8       # speed when spinning in place
STICK_DEADZONE = 0.12
KEY_DEADMAN_SEC = 0.6      # held key with no repeat events for this long -> treat as released
CONTROL_HZ = 50

# --- Ultrasonic tuning ----------------------------------------------------------
# Note: classic HC-SR04 is a 5 V part; powered from 3.3 V many units read nothing or
# are erratic. If so, use 5 V + 1k/2k divider on Echo, or a 3.3 V-capable HC-SR04P / RCWL-1601.
ULTRASONIC_MAX_CM = 300            # readings beyond this = "nothing there"
ULTRASONIC_TIMEOUT_S = 0.03        # ~5 m round trip
ULTRASONIC_GAP_S = 0.012           # pause between sensors to avoid cross-talk
OBSTACLE_STOP_CM = 10              # every move stops when a sensor in the direction of travel is closer than this
SLOW_ZONE_CM = 35                  # ...and slows down to AUTO_MIN_SPEED once something is closer than this

# Mounting: center faces straight ahead, the two front corners are angled 30 deg outward,
# rear faces straight back. Degrees: 0 = ahead, + = right, 180 = behind.
ULTRASONIC_ANGLES = {"front_left": -30, "front_center": 0, "front_right": 30, "rear": 180}
# Which sensors guard which motion (computed from the angles at the bottom of this file unless a
# robot's local_config.py sets them): forward = sensors facing forward, backward = facing back,
# turning left/right = sensors angled to that side (spinning swings the front corners out first).
STOP_CM_FORWARD = None
STOP_CM_BACKWARD = None
STOP_CM_TURN = None

# --- Autonomous motion (LLM tool calls) ------------------------------------------
# No wheel encoders, so distance/angle are timed. CALIBRATE: drive 1 s at speed 1.0 and
# measure metres; spin 1 s at speed 1.0 and measure degrees.
METRES_PER_SEC_AT_FULL = 0.5
DEGREES_PER_SEC_AT_FULL = 180.0
AUTO_DEFAULT_SPEED = 0.6
AUTO_MIN_SPEED = 0.35               # used near obstacles (SLOW_ZONE_CM)
AUTO_MAX_SPEED = 0.8
AUTO_MAX_MOVE_SEC = 6.0            # hard cap per single command
AUTO_MAX_DISTANCE_M = 3.0

# --- Voice agent ------------------------------------------------------------------
LLAMA_SERVER = "/home/maximus/src/llama.cpp/build/bin/llama-server"
# One model at a time does everything (chat, tool calls / driving, debates). Slots each keep their
# own cached prompt: 0 = chat + tools, 1 = debating a person, 2 = debating the other robot.
# Several "brains" can be installed; the BRAIN button on the face (or the B key, or "switch your
# brain") swaps between them. The choice is remembered in logs/brain.txt.
BRAINS = {
    "qwen": {"label": "Qwen3 4B", "model": "models/llm/Qwen3-4B-Instruct-2507-Q4_0.gguf",
             "port": 8082, "ctx": 12288, "slots": 3},
    "lfm":  {"label": "LFM3 Omni 2B", "model": "models/llm/LFM3-Omni-2B-Q4_0.gguf",
             "port": 8082, "ctx": 12288, "slots": 3},
}
BRAIN = "lfm"                      # default brain (used when logs/brain.txt doesn't exist yet)
LLM = BRAINS[BRAIN]                # (kept for older code; the running brain is in rover/llm.py)
SLOT_CHAT, SLOT_DEBATE, SLOT_ROBOT_DEBATE = 0, 1, 2
LLM_THREADS_GEN = 3                # 3 beats 4 for generation on Pi 5 (memory bound)
LLM_THREADS_BATCH = 4
HISTORY_TURNS = 8                  # user+assistant pairs kept in context

TTS_VOICE = "models/piper/en_US-lessac-medium.onnx"   # Lyra: female. Nova sets a male voice in its local_config.py
TTS_SPEED = 0.9                    # >1 slower, <1 faster (Piper length_scale); chatting is a bit quicker than reading
TTS_EXPRESSION = 0.8               # pitch/intonation variation (Piper noise_scale; voice default 0.667)
TTS_VARIETY = 0.05                 # random tempo/pause drift per sentence so it sounds talked, not read (0 = off)
TTS_PAUSE_S = 0.22                 # breathing gap between sentences (x1.2 after ? and !, half after "Oh!")
TTS_PAUSE_CLAUSE_S = 0.12          # gap after a piece cut at a comma (normal chat only)
TTS_GAIN = 3.0                     # loudness boost via soft limiter (1 = off, 3 = ~+7 dB, 4 = ~+8 dB, rougher)
TTS_RHYTHM = 1.0                   # timing variation between sounds (Piper noise_w; voice default 0.8)
CONTINUATION_WAIT_S = 1.5          # unfinished-sounding phrase: wait this long for the rest
POST_SPEECH_MUTE_S = 0.6           # keep the mic muted briefly after talking (Bluetooth delay + echo tail)

ROBOT_NAME = "Lyra"                # each robot sets its own in rover/local_config.py
# Wake word: in normal conversation the robot only answers when its name is in the sentence,
# then keeps listening without the name for WAKE_WINDOW_S after each reply. (Debate mode with a
# person needs no wake word.) Start with --no-wake-word to answer everything.
WAKE_WORD = True
WAKE_WORDS = None                  # None = the robot's name + common mishearings of it
WAKE_WINDOW_S = 20

# --- Audio device -------------------------------------------------------------------
# Bluetooth speaker+mic to keep connected (None = use whatever PipeWire defaults to).
BT_AUDIO_MAC = "41:42:04:5B:3E:70"   # "Pico" speaker (was AC:EF:92:E6:D6:1A, Xiaomi Sound Outdoor)
BT_MUSIC_WHILE_SPEAKING = True       # speak in music mode (A2DP: louder, clearer); the mic is closed while talking
MIC_SILENT_TIMEOUT_S = 3.0           # no audio data at all for this long -> mic is gone

# --- Turn-taking ---------------------------------------------------------------------
# Strictly one at a time: the mic is muted while the robot talks and anything heard while it is
# thinking is dropped (except "stop"). There is no talking over the robot.
PARTIAL_EVERY_S = 0.8                # live transcript refresh while you speak
STT_THREADS = 3                      # Parakeet threads: 3 is fastest on a Pi 5 (4 is no faster, and fights the LLM)
BUSY_MAX_PHRASE_S = 3.0              # while thinking/debating, skip phrases longer than this (only "stop" matters)
SAVE_UTTERANCES = True               # keep the last 30 heard phrases in logs/utterances/ (debugging STT)

# --- Robot vs robot debate -------------------------------------------------------------
# The two rovers talk over the network (same Wi-Fi) so each always gets the other's exact words;
# they still speak out loud for the audience. Set PEER_* on each robot in rover/local_config.py.
PEER_NAME = None                     # the other robot's name, e.g. "Nova"
PEER_HOST = None                     # its address, e.g. "nova.local" or "192.168.1.42"
LINK_PORT = 8765                     # both robots listen on this TCP port
PEER_PORT = None                     # the other robot's port if different (testing two on one Pi)
LINK_KEY = "change-me"               # shared secret, must be the same on both robots
ROBOT_DEBATE_ROUNDS = 3              # turns per robot (opening, rebuttal..., closing)
ROBOT_DEBATE_WORDS = 30              # max length of each turn (turns are one sentence)

# --- Per-robot overrides ------------------------------------------------------------------
# rover/local_config.py (not shared between robots) can override anything above: name, pins,
# sensors, calibration, peer address. See rover/local_config.example.py.
try:
    from .local_config import *  # noqa: F401,F403
except ImportError:
    pass


def _angle(name):
    if name in ULTRASONIC_ANGLES:
        return ULTRASONIC_ANGLES[name]
    return 180 if "rear" in name or "back" in name else 0


if STOP_CM_FORWARD is None:
    STOP_CM_FORWARD = {n: OBSTACLE_STOP_CM for n in ULTRASONIC if abs(_angle(n)) < 90}
if STOP_CM_BACKWARD is None:
    STOP_CM_BACKWARD = {n: OBSTACLE_STOP_CM for n in ULTRASONIC if abs(_angle(n)) > 90}
if STOP_CM_TURN is None:
    STOP_CM_TURN = {"left": {n: OBSTACLE_STOP_CM for n in ULTRASONIC if -90 < _angle(n) < 0},
                    "right": {n: OBSTACLE_STOP_CM for n in ULTRASONIC if 0 < _angle(n) < 90}}
# speech-recognition slips of each name (not real words/names, so they can't trigger by accident)
MISHEARD_NAMES = {"lyra": ["lira", "laira", "leira", "lyre", "lyrah"],
                  "nova": ["novah", "noha", "nohva", "nowa", "nowva", "nava", "novva", "nouva", "knova"],
                  "rover": ["rovers", "rova", "rober"]}
# when the OTHER robot is named in "debate <name>" a looser match is fine (the word "debate" is
# needed too), so a few real words that sound like it count as well
_PEER_EXTRA = {"nova": ["noah", "nover", "novo", "no va", "nor va", "nova's", "lova", "nora", "noa"], "lyra": ["laura", "lara", "lyrics"]}
if WAKE_WORDS is None:
    WAKE_WORDS = [ROBOT_NAME.lower()] + MISHEARD_NAMES.get(ROBOT_NAME.lower(), [])
PEER_NAMES = ([PEER_NAME.lower()] + MISHEARD_NAMES.get(PEER_NAME.lower(), []) + _PEER_EXTRA.get(PEER_NAME.lower(), [])
              if PEER_NAME else [])
