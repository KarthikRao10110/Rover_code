"""Rover voice agent: face + ears + brain + voice + wheels, all in one process.

    .venv/bin/python -m rover.main                 # everything, fullscreen face
    .venv/bin/python -m rover.main --windowed      # face in a window (testing on the desktop)
    .venv/bin/python -m rover.main --no-motors --no-sensors
Usually started via ./start_rover.sh or the desktop icon.
"""

import argparse
import os
import signal
import threading
import time

from . import config
from .agent import Agent, State
from .llm import ROOT, start_server, wait_ready

LOG_DIR = os.path.join(ROOT, "logs")


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windowed", action="store_true", help="face in a window instead of fullscreen")
    ap.add_argument("--no-face", action="store_true", help="run without any display")
    ap.add_argument("--face", choices=["raylib", "pygame"], default="raylib",
                    help="raylib = GPU face with expressions + lip sync (default); pygame = simple fallback")
    ap.add_argument("--no-motors", action="store_true")
    ap.add_argument("--no-sensors", action="store_true")
    ap.add_argument("--no-wake-word", action="store_true",
                    help=f'answer everything, not only when called "{config.ROBOT_NAME}"')
    args = ap.parse_args()
    os.chdir(ROOT)
    os.makedirs(LOG_DIR, exist_ok=True)

    state = State()
    procs = []
    parts = {}

    def shutdown(*_):
        if not state.alive:
            return
        log("shutting down")
        state.alive = False

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    def interrupt():
        if "agent" in parts:
            parts["agent"].interrupt("screen/space bar")

    # ------------------------------------------------------------------ boot ------
    def boot():
        try:
            state.info = "starting the language model..."
            llm = start_server(config.LLM, os.path.join(LOG_DIR, "llm.log"))
            if llm:
                procs.append(llm)

            state.info = "loading voice..."
            from .tts import Speaker
            speaker = Speaker()
            parts["speaker"] = speaker

            state.info = "connecting speaker..."
            from .audio import ensure_bt, mic_target
            if not ensure_bt(log):
                state.info = "speaker/mic not connected - turn it on, retrying..."

            state.info = "loading ears..."
            from .stt import Listener, ListenerConfig
            listener = Listener(ListenerConfig(mic_target=mic_target(),
                                               silent_timeout=config.MIC_SILENT_TIMEOUT_S,
                                               save_dir=os.path.join(LOG_DIR, "utterances")
                                               if config.SAVE_UTTERANCES else None),
                                partial_every=config.PARTIAL_EVERY_S)
            parts["listener"] = listener

            sensors = None
            if not args.no_sensors:
                try:
                    from .sensors import UltrasonicArray
                    sensors = UltrasonicArray().start()
                except Exception as e:
                    log(f"[sensors] disabled: {e}")
            parts["sensors"] = sensors

            rover = None
            if not args.no_motors:
                try:
                    from .motion import RoverControl
                    rover = RoverControl(sensors=sensors)
                except Exception as e:
                    log(f"[motors] disabled: {e}")
            parts["rover"] = rover

            state.info = "waking up the brain..."
            wait_ready(config.LLM["port"], proc=llm)

            from .link import PeerLink
            link = PeerLink(on_message=lambda m: parts["agent"].robot_debate.on_message(m), log=log)
            try:
                link.start()
            except OSError as e:
                log(f"[link] robot-to-robot link disabled: {e}")
                link = None
            agent = Agent(state, speaker, rover, sensors, link=link,
                          wake_word=config.WAKE_WORD and not args.no_wake_word, log=log)
            parts["agent"] = agent
            state.info = "warming up the brain (first boot takes ~40 s)..."
            agent.warm_up()

            # live transcript while you talk (not while the robot is thinking: it would steal CPU)
            listener.on_partial = lambda t: setattr(state, "user_text", t + " …")
            listener.partials_enabled = lambda: not agent.busy() and not agent.robot_debate.active
            # during a robot debate only short phrases ("stop the debate") are worth transcribing
            listener.max_phrase_s = lambda: 3.0 if agent.robot_debate.active else None

            # strict turn-taking: the mic is muted while the robot talks (+ a short echo tail)
            unmute_timer = [None]

            def on_speaking(active):
                if unmute_timer[0]:
                    unmute_timer[0].cancel()
                if active:
                    listener.muted = True
                else:
                    unmute_timer[0] = threading.Timer(config.POST_SPEECH_MUTE_S,
                                                      lambda: setattr(listener, "muted", False))
                    unmute_timer[0].start()
            speaker.on_state = on_speaking

            def listen_loop():
                while state.alive:
                    try:
                        for text in listener.utterances():
                            if not state.alive:
                                return
                            if state.info.startswith("mic"):
                                state.info = ""
                            agent.hear(text)
                    except Exception as e:  # BT dropped / mic unplugged: reconnect and retry
                        log(f"[mic] {e}")
                        state.info = "mic disconnected - reconnecting..."
                        if ensure_bt(log):
                            state.info = ""
                        else:
                            time.sleep(3)

            threading.Thread(target=listen_loop, name="listen", daemon=True).start()
            threading.Thread(target=agent.run, name="agent", daemon=True).start()
            log(f"ready  (motors: {'yes' if rover else 'no'}, sensors: {'yes' if sensors else 'no'}, "
                f"wake word: {'yes' if agent.wake_word else 'no'}, "
                f"other robot: {config.PEER_NAME + ' @ ' + config.PEER_HOST if link and link.configured() else 'not set'})")
            speaker.say(f"Hi, I'm {config.ROBOT_NAME}. I'm awake and listening.")
            agent.warm_up(background=True)       # debate prompts: ready by the time anyone debates
        except Exception as e:
            log(f"[boot] FAILED: {type(e).__name__}: {e}")
            state.info = f"startup failed: {e}"

    # ---------------------------------------------------- live status for the face --
    def status_loop():
        while state.alive:
            sp, li, ag, ro = parts.get("speaker"), parts.get("listener"), parts.get("agent"), parts.get("rover")
            if ag:
                if sp and sp.speaking():
                    state.status = "speaking"
                elif ag.busy():
                    state.status = "thinking"
                elif li and not li.muted and li.hearing_speech:
                    state.status = "listening"
                else:
                    state.status = "idle"
            state.mouth = sp.mouth_level() if sp else 0.0
            state.viseme = sp.viseme() if sp and hasattr(sp, "viseme") else "rest"
            if sp:
                state.reply_text = sp.caption()   # subtitles follow the voice, not the LLM
            state.mic_level = li.level if li and not li.muted else 0.0
            state.moving = bool(ro and ro.busy())
            time.sleep(1 / 30)

    threading.Thread(target=boot, name="boot", daemon=True).start()
    threading.Thread(target=status_loop, name="status", daemon=True).start()

    try:
        if args.no_face:
            while state.alive:
                time.sleep(0.2)
        else:
            face = None
            if args.face == "raylib":
                try:
                    from .face_rl import FaceRL
                    face = FaceRL(state, on_interrupt=interrupt, on_quit=shutdown, fullscreen=not args.windowed)
                except Exception as e:
                    log(f"[face] raylib face failed ({e}); using the simple face")
            if face is None:
                from .face import Face
                face = Face(state, on_interrupt=interrupt, on_quit=shutdown, fullscreen=not args.windowed)
            face.run()
    finally:
        state.alive = False
        for name in ("rover", "sensors"):
            p = parts.get(name)
            if p:
                try:
                    p.close()
                except Exception:
                    pass
        if parts.get("speaker"):
            parts["speaker"].cancel()
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(5)
            except Exception:
                p.kill()
        log("bye")
        os._exit(0)  # don't wait on the blocking mic thread


if __name__ == "__main__":
    main()
