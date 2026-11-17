"""Spin each side one at a time so you can check wiring and direction.

PUT THE ROVER ON A BOX WITH ALL WHEELS OFF THE GROUND FIRST.

    python3 -m tools.motor_test             # 50% duty, 1.5 s per step
    python3 -m tools.motor_test --duty 0.3  # find the lowest duty that still turns -> MIN_DUTY
"""

import argparse
import time

from rover import config
from rover.drive import Side


def run(name, side, duty, secs):
    for label, d in (("FORWARD", duty), ("BACKWARD", -duty)):
        print(f"{name:5s} {label:8s} at {duty:.0%} ...", flush=True)
        side.raw(d)
        time.sleep(secs)
        side.raw(0)
        time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duty", type=float, default=0.5)
    ap.add_argument("--secs", type=float, default=1.5)
    args = ap.parse_args()

    left = Side(config.LEFT_IN1, config.LEFT_IN2, config.LEFT_EN, config.LEFT_INVERTED)
    right = Side(config.RIGHT_IN3, config.RIGHT_IN4, config.RIGHT_EN, config.RIGHT_INVERTED)
    try:
        input("Wheels off the ground? Press Enter to start (Ctrl+C to abort)...")
        run("LEFT", left, args.duty, args.secs)
        run("RIGHT", right, args.duty, args.secs)
        print("\nIf a side spun the wrong way on FORWARD, set LEFT_INVERTED / RIGHT_INVERTED in rover/config.py")
    finally:
        left.close()
        right.close()


if __name__ == "__main__":
    main()
