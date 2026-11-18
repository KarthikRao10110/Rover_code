"""Deterministic parser for common driving commands — instant and reliable, no LLM.

    parse("move forward 2 meters and then turn left") ->
        [("move", {"direction": "forward", "distance_m": 2.0}), ("turn", {"direction": "left", "degrees": 90})]
Returns None if the sentence isn't a plain driving command (the LLM handles it instead).
"""

import re

NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
    "forty-five": 45, "fifty": 50, "sixty": 60, "ninety": 90, "half a": 0.5, "half": 0.5,
    "a half": 0.5, "a quarter": 0.25, "quarter": 0.25, "a couple of": 2, "a couple": 2, "a few": 3,
}
NUM = r"(\d+(?:\.\d+)?|" + "|".join(sorted(map(re.escape, NUM_WORDS), key=len, reverse=True)) + r")"
UNIT_M = {"m": 1, "meter": 1, "meters": 1, "metre": 1, "metres": 1, "cm": 0.01, "centimeter": 0.01,
          "centimeters": 0.01, "centimetre": 0.01, "centimetres": 0.01, "foot": 0.3048, "feet": 0.3048,
          "ft": 0.3048, "step": 0.25, "steps": 0.25, "inch": 0.0254, "inches": 0.0254}
UNIT_S = {"s", "sec", "secs", "second", "seconds"}

FILLER = re.compile(r"\b(please|can you|could you|would you|will you|rover|lyra|nova|robot|hey|now|just|"
                    r"slowly|quickly|a bit|a little|for me|okay|ok|so|i'll|i will|let's|lets|you|we|"
                    r"i want you to|i need you to|go ahead and|try to)\b|[,.!?]", re.I)

# speech-recognition slips seen with the Bluetooth mic ("go friend" = "go forward")
MISHEARD = [
    (re.compile(r"\b(go|move|drive|going|moving|roll|head)\s+(friend|friends|front|for ward|for word|four word|"
                r"fort|forth|for what|four ward)\b", re.I), r"\1 forward"),
    (re.compile(r"\b(go|move|drive|going|moving)\s+(bag|buck|pack|bark|bad|beck)\b", re.I), r"\1 back"),
    (re.compile(r"\bturn\s+(write|rite|wright|ride|bright)\b", re.I), r"turn right"),
    (re.compile(r"\bturn\s+(lift|laughed|lef|loft)\b", re.I), r"turn left"),
]
# a command somewhere inside a short sentence ("I think we go back", "go forward go forward")
RE_CMD_ANY = re.compile(
    r"\b(?:(?:go|move|drive|head|roll)\s+(?:straight\s+)?(?:forward|forwards|ahead|straight|backward|backwards|back)"
    r"(?:\s+(?:by|for)?\s*\S+\s+(?:m|meters?|metres?|cm|centimet(?:er|re)s?|feet|foot|steps?|seconds?|secs?))?"
    r"|back up|reverse"
    r"|turn\s+(?:to\s+the\s+)?(?:left|right)(?:\s+\S+\s+degrees?)?"
    r"|turn\s+around|spin(?:\s+around)?|stop)\b", re.I)
SHORT_WORDS = 7


def normalize(text):
    for rx, rep in MISHEARD:
        text = rx.sub(rep, text)
    return text
SPLIT = re.compile(r"\b(?:and then|then|and|after that|followed by)\b", re.I)

RE_MOVE = re.compile(rf"^(?:go|move|drive|come|roll|head|back up|reverse)?\s*(?:straight\s*)?"
                     rf"(forward|forwards|ahead|straight|backward|backwards|back|in reverse)?\s*"
                     rf"(?:by|for)?\s*(?:{NUM}\s*([a-z]+))?$", re.I)
RE_TURN = re.compile(rf"^(?:turn|rotate|spin|face|look|go|steer)\s*(?:to\s*(?:the|your)?\s*)?(left|right)"
                     rf"(?:\s*(?:by)?\s*{NUM}\s*(?:degrees?|deg))?$", re.I)
RE_TURN_DEG_FIRST = re.compile(rf"^(?:turn|rotate|spin)\s*{NUM}\s*(?:degrees?|deg)\s*(?:to\s*(?:the)?\s*)?(left|right)?$", re.I)
RE_AROUND = re.compile(r"^(?:turn|spin|rotate)\s*(?:yourself\s*)?around$|^do a (?:u-?turn|one eighty|180)$|^u-?turn$", re.I)
RE_SPIN = re.compile(r"^(?:spin|do a spin|do a (?:full )?circle|spin around in a circle|do a 360|turn 360)$", re.I)
RE_STOP = re.compile(r"^(?:stop|halt|freeze|brake|stop moving|stay|stay there)$", re.I)
RE_LOOK = re.compile(r"(what('s| is) (in front|behind|around)|anything (is |there )?(in front|ahead|behind)|"
                     r"(check|read|scan) (your |the )?(sensors?|surroundings|distance)|how far (away )?(is|are)\b|"
                     r"obstacles?)", re.I)


def _num(s):
    s = s.lower().strip()
    return float(s) if re.match(r"^\d", s) else float(NUM_WORDS.get(s, 1))


TIMES = {"once": 1, "twice": 2, "two times": 2, "three times": 3, "thrice": 3, "3 times": 3, "2 times": 2}
RE_SPIN_N = re.compile(r"^(?:spin|turn|rotate)\s*(?:around\s*)?(?:in a circle\s*)?(once|twice|thrice|two times|three times|2 times|3 times)$", re.I)


def _one(part):
    p = re.sub(r"\s+", " ", FILLER.sub(" ", part)).strip().lower()
    if not p:
        return None
    m = RE_SPIN_N.match(p)
    if m:
        return ("turn", {"direction": "right", "degrees": 360 * TIMES[m.group(1).lower()]})
    if RE_STOP.match(p):
        return ("stop", {})
    if RE_AROUND.match(p):
        return ("turn", {"direction": "right", "degrees": 180})
    if RE_SPIN.match(p):
        return ("turn", {"direction": "right", "degrees": 360})
    m = RE_TURN.match(p)
    if m:
        return ("turn", {"direction": m.group(1), "degrees": _num(m.group(2)) if m.group(2) else 90})
    m = RE_TURN_DEG_FIRST.match(p)
    if m:
        return ("turn", {"direction": m.group(2) or "right", "degrees": _num(m.group(1))})
    m = RE_MOVE.match(p)
    if m and (m.group(1) or p.split()[0] in ("back", "reverse", "come")) and p.split()[0] in (
            "go", "move", "drive", "come", "roll", "head", "back", "reverse", "forward", "forwards", "ahead", "backward", "backwards"):
        word = (m.group(1) or "").lower()
        direction = "backward" if (word.startswith("back") or "reverse" in p or p.startswith("back")) else "forward"
        args = {"direction": direction}
        if m.group(2):
            n, unit = _num(m.group(2)), (m.group(3) or "").lower()
            if unit in UNIT_M:
                args["distance_m"] = n * UNIT_M[unit]
            elif unit in UNIT_S:
                args["seconds"] = n
            else:
                return None
        else:
            args["distance_m"] = 0.5
        return ("move", args)
    return None


RE_COME = re.compile(r"^\W*(?:can you |could you |please |rover,? )*(come (here|over here|to me|back)|"
                     r"come (a (little|bit) )?closer|approach me)\b", re.I)
RE_AWAY = re.compile(r"^\W*(?:can you |could you |please |rover,? )*(go away|back off|move away|"
                     r"give me (some )?space|get back)\b", re.I)
RE_DANCE = re.compile(r"\b(dance|do a dance|wiggle|show me your moves|boogie)\b", re.I)
DANCE = [("turn", {"direction": "left", "degrees": 30}), ("turn", {"direction": "right", "degrees": 60}),
         ("turn", {"direction": "left", "degrees": 60}), ("turn", {"direction": "right", "degrees": 30}),
         ("move", {"direction": "forward", "distance_m": 0.15}), ("move", {"direction": "backward", "distance_m": 0.15}),
         ("turn", {"direction": "right", "degrees": 360})]


def parse(text):
    """List of (tool, args) if the sentence is driving commands, else None."""
    text = normalize(text)
    if RE_LOOK.search(text):
        return [("read_distances", {})]
    if RE_DANCE.search(text):
        return [("dance", {})] + DANCE
    if RE_COME.search(text):
        far = not re.search(r"\bcloser\b", text, re.I)
        return [("move", {"direction": "forward", "distance_m": 1.0 if far else 0.4})]
    if RE_AWAY.search(text):
        return [("move", {"direction": "backward", "distance_m": 0.5})]
    parts = [x for x in SPLIT.split(text) if x and x.strip()]
    out = []
    for part in parts:
        cmd = _one(part)
        if cmd is None:
            out = None
            break
        out.append(cmd)
    if out:
        return out
    return _scan(text)


def _scan(text):
    """Short utterance with a command buried in it: pull the commands out in order."""
    words = re.sub(r"\s+", " ", FILLER.sub(" ", text)).split()
    if not words or len(words) > SHORT_WORDS:
        return None
    out = []
    for m in RE_CMD_ANY.finditer(text):
        cmd = _one(m.group(0))
        if cmd and (not out or out[-1] != cmd):   # "go forward go forward" -> once
            out.append(cmd)
    return out or None


def describe(cmd):
    """Short spoken acknowledgement for a parsed command."""
    name, a = cmd
    if name == "dance":
        return "Watch this!"
    if name == "stop":
        return "Stopping."
    if name == "turn":
        d = a.get("degrees", 90)
        if d >= 720:
            return f"Spinning around {d / 360:g} times!"
        if d >= 360:
            return "Spinning around!"
        if d == 180:
            return "Turning around."
        return f"Turning {a['direction']}" + (f" {d:g} degrees." if d != 90 else ".")
    if name == "move":
        if "seconds" in a:
            return f"Going {a['direction']} for {a['seconds']:g} seconds."
        m = a.get("distance_m", 0.5)
        dist = f"{m * 100:.0f} centimetres" if m < 1 else f"{m:g} metre" + ("s" if m != 1 else "")
        return f"Going {a['direction']} {dist}."
    return ""


if __name__ == "__main__":
    tests = ["Move forward one meter.", "Turn around.", "go back 50 centimeters", "turn left",
             "Rover, turn right 45 degrees", "move forward 2 meters and then turn left", "spin",
             "Can you check if anything is in front of you?", "come forward", "drive forward for 3 seconds",
             "What is the capital of Japan?", "Go to the kitchen", "turn 90 degrees left", "back up",
             "I think we should move forward with the plan", "go forward two steps then turn right and stop"]
    for t in tests:
        r = parse(t)
        print(f"{t!r:55} -> {r}" + (f"   say: {[describe(c) for c in r if c[0] != 'read_distances']}" if r else ""))
