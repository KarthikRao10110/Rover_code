"""Ultrasonic sensor tests.

    python3 -m tools.sensor_test read                 # live readout of all 4 sensors + health report
    python3 -m tools.sensor_test read --secs 10       # stop by itself after 10 s
    python3 -m tools.sensor_test drive                # drive forward, stop when something is < 5 cm
    python3 -m tools.sensor_test drive --stop-cm 15 --speed 0.3
    python3 -m tools.sensor_test drive --direction backward   # uses the rear sensor

Run in a normal terminal (not through Claude's "!" prompt) so you can see it live.
Ctrl+C always stops the motors.

Drive test safety:
  * refuses to start unless every sensor it relies on is responding (open space is fine)
  * stops if a sensor that was reading close suddenly reads nothing (an HC-SR04 goes blind
    under ~2 cm, so "no echo" right after "very close" means "touching")
  * stops after --max-secs no matter what
"""

import argparse
import sys
import time

from rover import config
from rover.sensors import UltrasonicArray

FRONT = list(config.STOP_CM_FORWARD)          # works for the 4-sensor and the 2-sensor rover
REAR = list(config.STOP_CM_BACKWARD)
LABEL = {n: n.upper().replace("_", "-") for n in config.ULTRASONIC}


def bar(cm, width=24, full=100.0):
    if cm is None:
        return " " * width
    n = int(min(cm, full) / full * width)
    return "#" * n + "." * (width - n)


def fmt(cm):
    return f"{cm:6.1f} cm" if cm is not None else "  no echo"


# ------------------------------------------------------------------ read ---------
def read_test(secs):
    arr = UltrasonicArray()
    names = [s.name for s in arr.sensors]
    stats = {n: {"ok": 0, "miss": 0, "min": None, "max": None, "sum": 0.0} for n in names}
    live = sys.stdout.isatty()
    print("Reading sensors" + (f" for {secs:.0f} s" if secs else " (Ctrl+C to stop)") +
          ". Wave your hand in front of each one.\n")
    if live:
        print("\n" * len(names), end="")
    start = time.monotonic()
    try:
        while not secs or time.monotonic() - start < secs:
            row = {}
            for s in arr.sensors:
                cm = arr.ping(s)
                row[s.name] = cm
                st = stats[s.name]
                if cm is None:
                    st["miss"] += 1
                else:
                    st["ok"] += 1
                    st["sum"] += cm
                    st["min"] = cm if st["min"] is None else min(st["min"], cm)
                    st["max"] = cm if st["max"] is None else max(st["max"], cm)
                time.sleep(config.ULTRASONIC_GAP_S)
            if live:
                sys.stdout.write(f"\033[{len(names)}A")
                for n in names:
                    sys.stdout.write(f"\033[K  {LABEL[n]:13} {fmt(row[n])}  |{bar(row[n])}|\n")
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        arr.close()

    print("\nHealth report")
    print(f"  {'sensor':13} {'echoes':>8} {'success':>8} {'min':>9} {'avg':>9} {'max':>9}   verdict")
    all_ok = True
    for n in names:
        st = stats[n]
        total = st["ok"] + st["miss"]
        rate = st["ok"] / total if total else 0
        avg = st["sum"] / st["ok"] if st["ok"] else None
        if st["ok"] == 0:
            verdict = "DEAD - no echo at all (wiring / power / pin?)"
        elif rate < 0.5:
            verdict = "FLAKY - mostly missing (loose wire, or open space > 3 m)"
        elif st["max"] - st["min"] < 0.5 and total > 20:
            verdict = "ok (reading never changed - did you wave at it?)"
        else:
            verdict = "ok"
        all_ok &= st["ok"] > 0
        f = lambda v: f"{v:7.1f}cm" if v is not None else "       --"
        print(f"  {LABEL[n]:13} {st['ok']:>8} {rate:>7.0%} {f(st['min'])} {f(avg)} {f(st['max'])}   {verdict}")
    trig_echo = ", ".join(f"{LABEL[n]} trig GPIO{t}/echo GPIO{e}" for n, (t, e) in config.ULTRASONIC.items())
    print(f"\n  pins: {trig_echo}")
    return all_ok


# ------------------------------------------------------------------ drive --------
def brake(drive):
    """Active brake: both inputs high with the bridge enabled shorts the motors."""
    for side in (drive.left, drive.right):
        side.fwd.on()
        side.back.on()
        side.en.value = 1
        side.current = 0.0
    time.sleep(0.25)
    drive.stop()


def drive_test(direction, stop_cm, speed, max_secs):
    from rover.drive import Drivetrain

    watch = FRONT if direction == "forward" else REAR
    sign = 1 if direction == "forward" else -1
    arr = UltrasonicArray()
    sensors = [s for s in arr.sensors if s.name in watch]
    # only look ~1.3 m ahead so each ping returns quickly -> faster reaction
    config.ULTRASONIC_TIMEOUT_S = 0.008

    print(f"Checking {', '.join(LABEL[n] for n in watch)} before moving...")
    # alive = the sensor answers at all (a sensor facing open space still sends a timeout pulse)
    dead = [LABEL[s.name] for s in sensors if not arr.alive(s)]
    if dead:
        arr.close()
        print(f"\nNOT DRIVING: {', '.join(dead)} not responding at all (wiring / power).")
        print("Check with:  python3 -m tools.sensor_test read")
        return False
    last = {s.name: None for s in sensors}
    for _ in range(6):
        for s in sensors:
            cm = arr.ping(s)
            if cm is not None:
                last[s.name] = cm
            time.sleep(config.ULTRASONIC_GAP_S)
    for n in watch:
        print(f"  {LABEL[n]:13} {fmt(last[n]) if last[n] is not None else 'alive, nothing within ~1.3 m'}")
    if all(v is None for v in last.values()):
        print(f"  (nothing in range: it will just drive for {max_secs:g} s unless you put something in its way)")
    near = [(n, v) for n, v in last.items() if v is not None and v < stop_cm]
    if near:
        arr.close()
        print(f"\nNOT DRIVING: already closer than {stop_cm:g} cm ({LABEL[near[0][0]]} {near[0][1]:.1f} cm). Move it back.")
        return False

    drive = Drivetrain()
    print(f"\nDriving {direction} at speed {speed:g}; will stop when anything is closer than {stop_cm:g} cm.")
    print("Ctrl+C to stop.  Starting in 3 s...", flush=True)
    reason = "?"
    try:
        time.sleep(3)
        drive.set_target(sign * speed, sign * speed)
        t0 = prev = time.monotonic()
        while True:
            now = time.monotonic()
            drive.update(now - prev)
            prev = now
            if now - t0 > max_secs:
                reason = f"time limit ({max_secs:g} s) reached with nothing closer than {stop_cm:g} cm"
                break
            hit = None
            for s in sensors:
                cm = arr.ping(s)
                if cm is not None and cm < stop_cm:
                    hit = f"{LABEL[s.name]} sees an obstacle at {cm:.1f} cm"
                elif cm is None and last[s.name] is not None and last[s.name] < max(15.0, 2 * stop_cm):
                    hit = f"{LABEL[s.name]} lost its echo right after reading {last[s.name]:.1f} cm (too close to measure)"
                if cm is not None:
                    last[s.name] = cm
                if hit:
                    break
                time.sleep(0.004)
            if hit:
                brake(drive)
                reason = hit
                break
            line = "  ".join(f"{LABEL[n]} {fmt(last[n])}" for n in watch)
            sys.stdout.write(f"\r\033[K  {now - t0:4.1f}s  {line}")
            sys.stdout.flush()
    except KeyboardInterrupt:
        reason = "Ctrl+C"
    finally:
        drive.stop()
        drive.close()
        time.sleep(0.4)  # let it settle, then report where it ended up
        final = {}
        for s in sensors:
            final[s.name] = arr.ping(s)
            time.sleep(config.ULTRASONIC_GAP_S)
        arr.close()
    print(f"\n\nSTOPPED: {reason}")
    print("Resting distance: " + ",  ".join(f"{LABEL[n]} {fmt(v)}" for n, v in final.items()))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read", help="live readout + health report")
    r.add_argument("--secs", type=float, default=0, help="stop after this many seconds (0 = until Ctrl+C)")
    d = sub.add_parser("drive", help="drive until an obstacle is close, then stop")
    d.add_argument("--direction", choices=["forward", "backward"], default="forward")
    d.add_argument("--stop-cm", type=float, default=float(config.OBSTACLE_STOP_CM))
    d.add_argument("--speed", type=float, default=0.25, help="0..1 (default 0.25 = slow)")
    d.add_argument("--max-secs", type=float, default=8.0)
    args = ap.parse_args()
    if args.cmd == "read":
        ok = read_test(args.secs)
    else:
        ok = drive_test(args.direction, args.stop_cm, max(0.05, min(1.0, args.speed)), args.max_secs)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
