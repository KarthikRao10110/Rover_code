"""Robot vs robot debate, coordinated over the network (rover/link.py).

The host (the robot you asked) picks the sides, tells the other robot, announces the topic and
opens. Turns alternate: host opening, guest opening, rebuttals..., host closing, guest closing,
then the host wraps up. Each robot has ROBOT_DEBATE_ROUNDS turns.

Only one robot talks at a time, and nobody waits longer than needed:
  1. my reply streams out of the LLM;
  2. as soon as it's fully written I send the text ("turn") so the other robot can start
     thinking about its answer while I'm still speaking;
  3. when I finish speaking I send "spoken" - only then does the other robot start talking.
The exact text goes over the network, so speech recognition errors can't derail the debate.
"""

import re
import threading
import time

from . import config
from .tts import clean_for_speech

TURN_TIMEOUT_S = 240      # no word from the other robot for this long -> give up


def other(side):
    return "against" if side == "for" else "for"


class RobotDebate:
    def __init__(self, agent):
        self.agent = agent
        self.active = False
        self.thinking = False          # generating a turn (the face shows "thinking")
        self.topic = ""
        self.side = ""
        self.rounds = config.ROBOT_DEBATE_ROUNDS
        self.host = False
        self.history = []
        self._gen = 0                  # bumps on stop(); stale threads quit
        self._cond = threading.Condition()
        self._spoken = -1              # highest turn number the other robot finished speaking
        self._last_heard = 0.0
        self._lock = threading.Lock()

    @property
    def peer(self):
        return config.PEER_NAME or "the other robot"

    @property
    def total(self):
        return 2 * self.rounds

    def log(self, msg):
        self.agent.log(f"[robot debate] {msg}")

    # ---- starting / stopping ---------------------------------------------------------
    def start_as_host(self, topic, side="for", rounds=None):
        link = self.agent.link
        if not (link and link.configured()):
            return "error: I don't know where the other robot is. Set PEER_NAME and PEER_HOST in local_config.py."
        topic = (topic or "").strip(" .?!")
        if not topic:                      # (voice requests are confirmed in agent.py first)
            return "error: I need a topic first. What should we debate about?"
        if self.active:
            return "error: We're already debating."
        side = side if side in ("for", "against") else "for"
        rounds = int(rounds or config.ROBOT_DEBATE_ROUNDS)
        reply = link.send({"type": "start", "topic": topic, "side": other(side), "rounds": rounds,
                           "from": config.ROBOT_NAME})
        if not reply:
            return f"error: I can't reach {self.peer}. Is it switched on and on the same Wi-Fi?"
        if not reply.get("ok"):
            return f"error: {self.peer} can't debate right now ({reply.get('error', 'busy')})."
        self._begin(topic, side, rounds, host=True)
        sp = self.agent.speaker
        sp.new_turn()
        sp.say(f"Debate time! Today's topic is: {topic}. I will argue {side}, and {self.peer} will argue "
               f"{other(side)}. I'll start.")
        self._spawn(0, None)
        return f"ok: debate started, I argue {side}"

    def _begin(self, topic, side, rounds, host):
        a = self.agent
        if a.rover:
            a.rover.stop()
        if not host and a.busy():          # the other robot started it: drop what I was doing
            a.interrupt("robot debate starting")
        with self._lock:
            self._gen += 1
            self.active, self.topic, self.side, self.rounds, self.host = True, topic, side, rounds, host
            self.history = []
        with self._cond:
            self._spoken = -1
        self._last_heard = time.monotonic()
        st = a.state
        st.mode = "debate"
        st.debate_topic = f"{topic}  ({config.ROBOT_NAME}: {side.upper()})"
        st.robot_side = side
        st.user_text = ""
        a._update_hint()
        self.log(f"started as {'host' if host else 'guest'}: '{topic}', my side {side}, {rounds} rounds")
        threading.Thread(target=self._watchdog, args=(self._gen,), name="debate-watchdog", daemon=True).start()

    def stop(self, notify_peer=False, reason="stopped"):
        if not self.active:
            return
        with self._lock:
            self._gen += 1
            self.active = False
            self.thinking = False
        with self._cond:
            self._cond.notify_all()
        a = self.agent
        a.llm.cancel()
        if notify_peer and a.link:
            threading.Thread(target=a.link.send, args=({"type": "end", "reason": reason},),
                             kwargs={"tries": 2}, daemon=True).start()
        st = a.state
        st.mode = "normal"
        st.debate_topic = st.robot_side = ""
        a._update_hint()
        self.log(f"ended ({reason})")

    def _watchdog(self, gen):
        while self.active and gen == self._gen:
            time.sleep(2)
            if self.agent.speaker.speaking() or self.thinking:
                self._last_heard = time.monotonic()
            elif time.monotonic() - self._last_heard > TURN_TIMEOUT_S:
                self.log("no reply from the other robot, giving up")
                self.agent.speaker.say(f"I haven't heard back from {self.peer}, so let's end the debate here.")
                self.stop(notify_peer=True, reason="timeout")

    # ---- messages from the other robot (link thread) ------------------------------------
    def on_message(self, msg):
        kind = msg.get("type")
        if kind == "host":            # tools/debate.py on this machine: start a debate
            r = self.agent.start_robot_debate(msg.get("topic", ""), msg.get("side", "for"))
            return {"ok": r.startswith("ok"), "message": r}
        if kind == "stop":            # tools/debate.py --stop
            self.agent.interrupt("debate stopped from the command line")
            return {"ok": True}
        if kind == "start":
            if self.active:
                return {"ok": False, "error": "already debating"}
            if self.agent.switching:
                return {"ok": False, "error": "switching brains, try again in a minute"}
            self._begin(msg.get("topic", ""), msg.get("side", "against"),
                        int(msg.get("rounds") or config.ROBOT_DEBATE_ROUNDS), host=False)
            return {"ok": True}
        if not self.active:
            return {"ok": False, "error": "not debating"}
        self._last_heard = time.monotonic()
        if kind == "turn":
            t = int(msg.get("turn", 0))
            text = msg.get("text", "")
            self.agent.state.user_text = f"{self.peer}: {text}"
            if t + 1 < self.total:
                self._spawn(t + 1, text)
        elif kind == "spoken":
            t = int(msg.get("turn", 0))
            with self._cond:
                self._spoken = max(self._spoken, t)
                self._cond.notify_all()
            if t == self.total - 1 and self.host:
                threading.Thread(target=self._wrap_up, daemon=True).start()
        elif kind == "end":
            if msg.get("reason") != "finished":
                self.agent.speaker.cancel()
                self.agent.speaker.say(f"{self.peer} stopped the debate.")
            self.stop(reason=msg.get("reason", "ended by the other robot"))
        return {"ok": True}

    # ---- taking a turn -----------------------------------------------------------------
    def _spawn(self, t, opponent_text):
        threading.Thread(target=self._take_turn, args=(t, opponent_text, self._gen),
                         name=f"debate-turn-{t}", daemon=True).start()

    def _part(self, t):
        k = t // 2                         # my own turn number (0-based)
        if k == 0:
            return "opening statement" + (f"; also answer what {self.peer} just said" if t == 1 else
                                          "; you speak first")
        if k >= self.rounds - 1:
            return "closing statement"
        return f"rebuttal {k} of {self.rounds - 2}"

    def _message(self, t, opponent_text):
        note = (f"[debate: topic \"{self.topic}\"; your side: {self.side.upper()}; {self.peer} argues "
                f"{other(self.side).upper()}; this turn: {self._part(t)}]")
        if opponent_text:
            return f"{self.peer} said: \"{opponent_text}\"\n{note}"
        return f"The moderator asks you to begin.\n{note}"

    def _may_speak(self, t, gen, timeout=None):
        """Wait until the other robot has finished speaking its previous turn."""
        with self._cond:
            return self._cond.wait_for(lambda: gen != self._gen or self._spoken >= t - 1, timeout)

    def _take_turn(self, t, opponent_text, gen):
        from .agent import split_sentences, RE_TOOL_LEAK, RE_END_OF_SENTENCE
        a, sp = self.agent, self.agent.speaker
        self.history.append({"role": "user", "content": self._message(t, opponent_text)})
        convo = self.history[-8:]
        while convo and convo[0]["role"] != "user":
            convo = convo[1:]
        messages = [{"role": "system", "content": a.system_prompt("robot_debate")}] + convo
        self.thinking = True
        pending, said, buf, first = [], [], "", True
        t0 = time.monotonic()
        finish = None

        def flush():
            if pending and gen == self._gen and self._spoken >= t - 1:
                if not said:
                    sp.new_turn()
                for s in pending:
                    sp.say(s)
                said.extend(pending)
                pending.clear()

        try:
            for kind, val in a.llm.stream(messages, slot=config.SLOT_ROBOT_DEBATE, temperature=0.8,
                                          max_tokens=int(config.ROBOT_DEBATE_WORDS * 2.5)):
                if gen != self._gen:
                    a.llm.cancel()
                    return
                if kind == "text":
                    buf += val
                    sentences, buf = split_sentences(buf, first, clauses=False)
                    done = False
                    for s in sentences:
                        if not RE_TOOL_LEAK.search(s):
                            pending.append(s)
                            first = False
                            if RE_END_OF_SENTENCE.search(s):
                                done = True       # debate turns are one sentence only
                                break
                    flush()
                    if done:
                        a.llm.cancel()
                        buf = ""
                        break
                elif kind == "done":
                    finish = (val.get("timings") or {}).get("finish_reason")
                    tm = val.get("timings") or {}
                    self.log(f"turn {t}: gen {tm.get('predicted_n', '?')} tok @ "
                             f"{tm.get('predicted_per_second', 0):.1f}/s in {time.monotonic() - t0:.0f}s")
        except Exception as e:
            self.log(f"LLM error: {e}")
        finally:
            self.thinking = False
        if gen != self._gen:
            return
        tail = buf.strip()
        if clean_for_speech(tail) and (re.search(r"[.!?]$", tail) or finish != "length"):
            pending.append(tail if re.search(r"[.!?]$", tail) else tail + ".")
        # a turn cut off by the token limit loses its unfinished last sentence (dropped above)
        text = " ".join(said + pending).strip() or "I'll let my previous points stand."
        if not said and not pending:
            pending.append(text)
        self.history.append({"role": "assistant", "content": text})
        self.log(f"turn {t} ({self._part(t)}): {text}")

        # let the other robot start thinking now, while I'm still talking
        if t + 1 < self.total and not a.link.send({"type": "turn", "turn": t, "text": text}):
            sp.say(f"I lost contact with {self.peer}, so let's stop the debate here.")
            self.stop(reason="lost contact")
            return
        # render my audio now, while the other robot is still talking, so I start the moment it stops
        clips = []
        if pending and self._spoken < t - 1:
            clips = [sp.synth(s) for s in pending]
        if not self._may_speak(t, gen, timeout=TURN_TIMEOUT_S) or gen != self._gen:
            return
        if clips:
            if not said:
                sp.new_turn()
            for c in clips:
                sp.play(c)
            said.extend(pending)
            pending.clear()
        flush()
        time.sleep(0.3)
        while sp.speaking() and gen == self._gen:
            time.sleep(0.1)
        if gen != self._gen:
            return
        time.sleep(0.4)                     # a natural pause before the other robot answers
        a.link.send({"type": "spoken", "turn": t})   # after the last turn the host wraps up

    def _wrap_up(self):
        if not self.active:
            return
        sp = self.agent.speaker
        while sp.speaking():
            time.sleep(0.1)
        sp.new_turn()
        sp.say(f"And that's the end of our debate on: {self.topic}. Thank you, {self.peer}, "
               f"and thank you all for listening. Now it's your turn to decide: who won?")
        self.agent.link.send({"type": "end", "reason": "finished"}, tries=2)
        self.stop(reason="finished")
