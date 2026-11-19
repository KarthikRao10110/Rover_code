# Rover

Raspberry Pi 5 rover → fully local voice agent with a face, driving, and debate mode.
Pin map and all tuning live in `rover/config.py`.

## Setup on a new Pi 5 (e.g. the second rover)
```bash
sudo apt install python3-lgpio python3-gpiozero python3-evdev python3-numpy python3-requests python3-pygame pipewire-bin
git clone https://github.com/KarthikRao10110/Rover_code.git rover_code && cd rover_code
python3 -m venv --system-site-packages .venv && .venv/bin/pip install -r requirements.txt
cp rover/local_config.example.py rover/local_config.py   # then edit: name, pins, sensors, other robot
```
Not in git (too big), put them in `models/`:
| file | source |
|---|---|
| `models/llm/Qwen3-4B-Instruct-2507-Q4_0.gguf` | Qwen3-4B-Instruct-2507 GGUF, Q4_0 (Hugging Face) |
| `models/llm/LFM3-Omni-2B-Q4_0.gguf` (optional 2nd brain) | Hugging Face `LiquidAI/LFM3-Omni-2B-GGUF`, Q4_0 |
| `models/parakeet-tdt-0.6b-v2/` (encoder/decoder/joiner `.int8.onnx`, `tokens.txt`) | sherpa-onnx release `sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8` |
| `models/silero_vad.onnx` | sherpa-onnx release `asr-models/silero_vad.onnx` |
| `models/piper/en_US-lessac-medium.onnx` (+ `.onnx.json`) | Hugging Face `rhasspy/piper-voices` (Lyra's voice, the default) |
| `models/piper/en_US-hfc_female-medium.onnx` (+ `.onnx.json`) | Hugging Face `rhasspy/piper-voices` (Nova's voice, set by `TTS_VOICE` in its `local_config.py`) |

```bash
# Nova's voice
cd models/piper && for e in onnx onnx.json; do
  curl -fLO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/hfc_female/medium/en_US-hfc_female-medium.$e; done
```

llama.cpp: build `llama-server` (`cmake -B build -DGGML_NATIVE=ON && cmake --build build -j4 --target llama-server`)
and set `LLAMA_SERVER` in `rover/config.py` if it isn't at `~/src/llama.cpp/build/bin/llama-server`.

## Start / stop
```bash
./start_rover.sh                 # everything, fullscreen face   (or double-click "Rover" on the desktop)
./start_rover.sh --windowed      # face in a window
./start_rover.sh --no-wake-word  # answer everything, not only when called by name
./start_rover.sh --help          # --no-motors --no-sensors --no-face
./stop_rover.sh
```
First start takes ~1–2 min (model load + prompt warm-up). Logs: `logs/rover.log`, `logs/llm.log`.

Face controls: **Space** or **tap the screen** = stop moving + stop talking (+ stop a robot debate),
**Esc/Q** quit, **F** fullscreen.

## Talking to it
- **Wake word:** say its name ("Lyra, what's photosynthesis?"). After each reply it keeps listening
  without the name for 15 s. In debate mode no name is needed.
- **Strict turn-taking:** you talk, it answers, then it listens again. The mic is muted while it
  speaks and anything said while it's thinking is ignored. Only **"stop"** always gets through.
- **Shut down:** "Lyra, shut down" (also "shutdown", "power off", "turn yourself off",
  "close the app") says bye and closes the whole app, like `./stop_rover.sh`.
- **Debates always confirm the topic first:** "start a debate" → "What should we debate about?" →
  you name it → "Just to check, you want to debate: … Shall we start?" → **yes** starts it,
  **no** (or "no, <other topic>") changes it, "never mind" cancels. Unclear answers (background
  noise) get re-asked; after 3 it gives up, and an unanswered question expires after 40 s.
- Plain driving commands ("go forward one metre", "turn left", "spin twice", "come here",
  "what's in front of you") are executed instantly without the LLM.
- Anything else goes to the LLM, which can drive itself with tools (below).

## How it works
```
mic → Silero VAD → Parakeet STT → wake word → router ─┬ "stop" ─────────→ motors off, voice off (instant)
                                                      ├ driving command → commands.py parser (instant)
                                                      ├ normal chat ────→ Qwen3-4B + tools        (slot 0)
                                                      ├ debate a person → Qwen3-4B, argues the other side (slot 1)
                                                      └ robot debate ───→ robot_debate.py + link.py   (slot 2)
LLM text streams → sentence splitter → Piper TTS → speaker (starts talking while still writing)
```
One model does everything (llama-server, port 8082, 3 cached slots). Two "brains" are installed and
you can swap between them while it runs: tap the **BRAIN** button in the top-right corner of the face,
press **B**, or say "Lyra, switch your brain". The choice is remembered (`logs/brain.txt`).

| brain | file | notes |
|---|---|---|
| Qwen3 4B (default) | `models/llm/Qwen3-4B-Instruct-2507-Q4_0.gguf` | best debates, ~10 s to first word |
| LFM3 Omni 2B | `models/llm/LFM3-Omni-2B-Q4_0.gguf` | much faster (~1 s to first word), weaker debater |

Get LFM3 Omni 2B (1 GB; needs llama.cpp v0.8.0 or newer):
```bash
curl -fL -o models/llm/LFM3-Omni-2B-Q4_0.gguf \
  https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF/resolve/main/LFM3-Omni-2B-Q4_0.gguf
```
Brains are listed in `BRAINS` in `rover/config.py`; one that isn't downloaded is simply skipped.

### LLM tools
| tool | what the model gets back |
|---|---|
| `move(direction, distance_m, speed)` | waits until done → `moved forward about 0.35 m; stopped early: obstacle 9 cm straight ahead; sensors now: ...` |
| `turn(direction, degrees)` | waits until done → `turned left about 90 degrees; finished; sensors now: ...` |
| `stop()` | stops immediately |
| `read_sensors()` | every sensor's distance + `can_move_forward/backward` with free room |
| `set_mode(mode, topic)` | switch to debating the user / back |
| `start_robot_debate(topic, my_side)` | only when the other robot is configured |

Every user message also carries a `[sensors: ahead 85 cm, front left clear, ...]` note, so the
model always knows its surroundings. Moves stop when anything in the direction of travel is
closer than **10 cm** (`OBSTACLE_STOP_CM`) and slow down inside 35 cm (`SLOW_ZONE_CM`).
Which sensors guard which motion is worked out from `ULTRASONIC_ANGLES`, so the same code runs
the 4-sensor rover and a front+rear 2-sensor rover.

## Two rovers debating (Lyra and Nova)
Both rovers run this same code; each has its own `rover/local_config.py` (name, motor pins,
sensors, the other robot's address). See `rover/local_config.example.py`.
They find each other by name (`PEER_HOST`, e.g. `raspberrypi.local`); if that fails, the rover scans
its Wi-Fi network for the other robot (port 8765 + the shared `LINK_KEY`), so a changed IP doesn't matter.
Speech-recognition slips of the names ("Noha", "Nohva", "Novah", "Noah"...) are in `MISHEARD_NAMES` in `rover/config.py`.
They talk over Wi-Fi (TCP port 8765), so each always gets the other's exact words, and they
still speak out loud for the audience. One talks at a time: while one is speaking, the other is
already writing its answer, and starts talking as soon as the first one finishes.

Start one:
- by voice: "Lyra, start pair debate" (also "robot debate", "debate each other", or
  "Lyra, debate Nova about whether homework should be banned"; add "you argue against" to swap
  sides). It asks for / repeats the topic and starts only after you say **yes**, or
- from a terminal on either robot: `.venv/bin/python -m tools.debate "Homework should be banned"`
  (`--side against`, `--stop`).

Each robot gets 3 turns (opening, rebuttal, closing). The host announces the topic and wraps up.
Stop it: say "stop the debate", tap the screen, or `tools.debate --stop`.

## Modules
| file | what |
|---|---|
| `rover/main.py` | wires everything together, boot sequence |
| `rover/agent.py` | brain: prompts, tools, routing, wake word, turn-taking |
| `rover/robot_debate.py` | robot vs robot debate (turn order, sides, timing) |
| `rover/link.py` | robot-to-robot messages over the network |
| `rover/local_config.py` | this robot's own settings (name, pins, sensors, peer) |
| `rover/commands.py` | deterministic driving-command parser |
| `rover/llm.py` | llama-server launcher + streaming client |
| `rover/stt.py` | always-on listener (VAD + Parakeet) |
| `rover/tts.py` | streaming Piper voice + mouth envelope |
| `rover/face_rl.py` | animated face (raylib, GPU): expressions + lip sync |
| `rover/face.py` | simple fallback face (pygame) |
| `rover/audio.py` | keeps the Bluetooth speaker/mic connected |
| `rover/motion.py` | timed autonomous moves with obstacle guards |
| `rover/sensors.py` | HC-SR04 array (any number), sequential pings via lgpio alerts |
| `rover/drive.py` | L298N skid-steer drivetrain |
| `rover/teleop.py` | manual keyboard / gamepad driving |

## Tools
```bash
python3 -m rover.teleop                 # manual driving (Phase 0)
python3 -m tools.motor_test             # wheels off the ground: direction check
.venv/bin/python -m rover.sensors       # live ultrasonic readout
.venv/bin/python -m tools.mic_test      # record 5 s + transcribe
.venv/bin/python -m tools.stt_check     # mic accuracy (word error rate)
.venv/bin/python -m tools.sensor_test read   # sensor check; `drive` = drive slowly until 10 cm
.venv/bin/python -m tools.debate "topic"  # start a robot debate
.venv/bin/python -m tools.listen        # live transcription
.venv/bin/python -m rover.commands      # command-parser self-test
```

## To calibrate once the rover is assembled
- `METRES_PER_SEC_AT_FULL`, `DEGREES_PER_SEC_AT_FULL` in `rover/config.py` (time 1 s at full speed).
- `MIN_DUTY` (lowest duty where wheels still turn).
- `OBSTACLE_STOP_CM` (10 cm) / `SLOW_ZONE_CM`. Check the robot body doesn't echo into the corner sensors.
