"""Timed autonomous moves for voice commands and LLM tool calls, with hard safety limits.

Each move runs in a background thread; `stop()` interrupts instantly. Sensors facing the
direction of travel stop the move when anything is closer than OBSTACLE_STOP_CM, and the
rover slows down once something is inside SLOW_ZONE_CM. After a move, `last_result` says
what actually happened (distance covered, why it stopped) so the LLM can react to it.
"""

import threading
import time

from . import config
from .drive import Drivetrain


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _where(name):
    from .sensors import UltrasonicArray
    return UltrasonicArray.label(name).split(" (")[0]


class RoverControl:
    def __init__(self, sensors=None):
        self.drive = Drivetrain()
        self.sensors = sensors
        self._cancel = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self.status = "idle"
        self.last_result = ""

    # ---- public API ------------------------------------------------------------
    def move(self, direction, distance_m=None, seconds=None, speed=None):
        """direction: forward | backward. Starts the move; returns "ok: ..." or "refused: ..."."""
        if direction not in ("forward", "backward"):
            return f"error: unknown direction '{direction}'"
        speed = _clamp(float(speed or config.AUTO_DEFAULT_SPEED), config.AUTO_MIN_SPEED, config.AUTO_MAX_SPEED)
        if distance_m is not None:
            distance_m = _clamp(float(distance_m), 0.05, config.AUTO_MAX_DISTANCE_M)
            seconds = distance_m / (config.METRES_PER_SEC_AT_FULL * speed)
        seconds = _clamp(float(seconds or 1.0), 0.1, config.AUTO_MAX_MOVE_SEC)
        guard = config.STOP_CM_FORWARD if direction == "forward" else config.STOP_CM_BACKWARD
        hit = self._blocked(guard)
        if hit:
            self.last_result = f"refused: obstacle {hit[1]:.0f} cm {_where(hit[0])}"
            return self.last_result
        sign = 1 if direction == "forward" else -1
        self._start(sign, sign, speed, seconds, f"moving {direction}", guard, metres=True)
        return f"ok: moving {direction} ~{config.METRES_PER_SEC_AT_FULL * speed * seconds:.2f} m"

    def turn(self, direction, degrees=90, speed=None):
        """direction: left | right. Spins in place."""
        if direction not in ("left", "right"):
            return f"error: unknown direction '{direction}'"
        speed = _clamp(float(speed or config.AUTO_DEFAULT_SPEED), config.AUTO_MIN_SPEED, config.AUTO_MAX_SPEED)
        degrees = _clamp(abs(float(degrees or 90)), 5, 1080)
        seconds = _clamp(degrees / (config.DEGREES_PER_SEC_AT_FULL * speed), 0.05, 2 * config.AUTO_MAX_MOVE_SEC)
        guard = config.STOP_CM_TURN.get(direction, {})
        hit = self._blocked(guard)
        if hit:
            self.last_result = f"refused: obstacle {hit[1]:.0f} cm {_where(hit[0])}"
            return self.last_result
        sign = 1 if direction == "right" else -1
        self._start(sign, -sign, speed, seconds, f"turning {direction}", guard, metres=False)
        return f"ok: turning {direction} ~{degrees:.0f} degrees"

    def stop(self):
        self._cancel.set()
        with self._lock:
            self.drive.stop()
        self.status = "idle"
        return "ok: stopped"

    def busy(self):
        return self._thread is not None and self._thread.is_alive()

    def wait(self, timeout=None):
        if self._thread:
            self._thread.join(timeout)

    def close(self):
        self.stop()
        self.wait(1)
        self.drive.close()

    # ---- internals -------------------------------------------------------------
    def _start(self, left, right, speed, seconds, status, guard, metres):
        self.stop()
        self.wait(1)
        self._cancel.clear()
        self.status = status
        self.last_result = ""
        self._thread = threading.Thread(target=self._run, args=(left, right, speed, seconds, status, guard, metres),
                                        name="motion", daemon=True)
        self._thread.start()

    def _blocked(self, guard):
        if not self.sensors or not guard:
            return None
        return self.sensors.blocked(guard)

    def _nearest(self, guard):
        if not self.sensors or not guard:
            return None
        d = [v for k, v in self.sensors.distances().items() if k in guard and v is not None]
        return min(d) if d else None

    def _run(self, left, right, speed, seconds, status, guard, metres):
        period = 1.0 / config.CONTROL_HZ
        start = last = time.monotonic()
        covered = 0.0          # speed-seconds actually driven (for the distance estimate)
        why = "finished"
        try:
            while not self._cancel.is_set():
                now = time.monotonic()
                if covered >= speed * seconds:
                    break
                hit = self._blocked(guard)
                if hit:
                    why = f"stopped early: obstacle {hit[1]:.0f} cm {_where(hit[0])}"
                    break
                near = self._nearest(guard)
                v = config.AUTO_MIN_SPEED if near is not None and near < config.SLOW_ZONE_CM else speed
                v = min(v, speed)
                with self._lock:
                    self.drive.set_target(left * v, right * v)
                    self.drive.update(now - last)
                covered += v * (now - last) if now > start else 0
                last = now
                time.sleep(period)
            else:
                why = "cancelled"
        finally:
            with self._lock:
                self.drive.stop()
            self.status = "idle"
        if metres:
            amount = f"{config.METRES_PER_SEC_AT_FULL * covered:.2f} m"
        else:
            amount = f"{config.DEGREES_PER_SEC_AT_FULL * covered:.0f} degrees"
        self.last_result = f"{status.replace('moving', 'moved').replace('turning', 'turned')} about {amount}; {why}"
