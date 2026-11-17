"""HC-SR04 ultrasonic array, read sequentially in a background thread.

Uses lgpio edge alerts (kernel timestamps, ns) instead of Python busy-waiting, so timing
stays accurate even while the LLM is hammering the CPU. Sensors fire one at a time so
their pings don't cross-talk.

    arr = UltrasonicArray(); arr.start()
    arr.distances()  -> {"front_left": 42.1, "front_center": None, ...}  (cm, None = no echo/clear)
    arr.min_front()  -> nearest front reading or None
"""

import threading
import time

import lgpio

from . import config

SPEED_OF_SOUND_CM_PER_NS = 34300 / 1e9


def _open_header_chip():
    """Pi 5 header GPIOs are on the RP1 chip (gpiochip0 on new kernels, gpiochip4 on old)."""
    for n in (0, 4):
        try:
            h = lgpio.gpiochip_open(n)
        except lgpio.error:
            continue
        if "rp1" in lgpio.gpio_get_chip_info(h)[3]:  # [status, lines, name, label]
            return h
        lgpio.gpiochip_close(h)
    raise RuntimeError("could not find the Pi 5 RP1 GPIO chip")


class _Sensor:
    def __init__(self, h, name, trig, echo):
        self.name, self.trig, self.echo = name, trig, echo
        self.rise = None
        self.fall = None
        self.done = threading.Event()
        self.busy_until = 0.0   # echo line may still be high from a ping that saw nothing
        lgpio.gpio_claim_output(h, trig, 0)
        lgpio.gpio_claim_alert(h, echo, lgpio.BOTH_EDGES)
        self.cb = lgpio.callback(h, echo, lgpio.BOTH_EDGES, self._edge)

    def _edge(self, chip, gpio, level, tick):
        if level == 1:
            self.rise = tick
        elif level == 0 and self.rise is not None:
            self.fall = tick
            self.done.set()


class UltrasonicArray:
    def __init__(self, sensors=None):
        self.h = _open_header_chip()
        spec = sensors or config.ULTRASONIC
        self.sensors = [_Sensor(self.h, n, t, e) for n, (t, e) in spec.items()]
        self._dist = {s.name: None for s in self.sensors}
        self._stamp = {s.name: 0.0 for s in self.sensors}
        self._ok = {s.name: False for s in self.sensors}  # ever got a valid echo?
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def _trigger(self, s):
        # An HC-SR04 that hears no echo holds its echo line high for ~40-50 ms and ignores
        # triggers until then, so never re-trigger a sensor while that may still be going on.
        wait = s.busy_until - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        s.rise = s.fall = None
        s.done.clear()
        lgpio.gpio_write(self.h, s.trig, 1)
        t = time.perf_counter()
        while time.perf_counter() - t < 12e-6:  # 10 us trigger pulse
            pass
        lgpio.gpio_write(self.h, s.trig, 0)
        return time.monotonic()

    def alive(self, s, tries=3):
        """True if the sensor answers at all - a real echo or its own "nothing in range"
        timeout pulse. Use this (not a distance) to tell a dead sensor from open space."""
        for _ in range(tries):
            self._trigger(s)
            if s.done.wait(0.25):
                time.sleep(0.01)
                return True
            s.busy_until = time.monotonic() + 0.06
        return False

    def ping(self, s):
        """One measurement in cm, or None if no echo / out of range."""
        t0 = self._trigger(s)
        if not s.done.wait(config.ULTRASONIC_TIMEOUT_S):
            s.busy_until = t0 + 0.06
            return None
        cm = (s.fall - s.rise) * SPEED_OF_SOUND_CM_PER_NS / 2
        if cm < 2 or cm > config.ULTRASONIC_MAX_CM:
            return None
        return cm

    def _run(self):
        while not self._stop.is_set():
            for s in self.sensors:
                cm = self.ping(s)
                with self._lock:
                    self._dist[s.name] = cm
                    self._stamp[s.name] = time.monotonic()
                    if cm is not None:
                        self._ok[s.name] = True
                time.sleep(config.ULTRASONIC_GAP_S)

    def start(self):
        self._thread = threading.Thread(target=self._run, name="ultrasonic", daemon=True)
        self._thread.start()
        return self

    def distances(self, max_age=0.5):
        now = time.monotonic()
        with self._lock:
            return {n: (d if now - self._stamp[n] <= max_age else None) for n, d in self._dist.items()}

    def working(self):
        """Names of sensors that have produced at least one valid echo."""
        with self._lock:
            return [n for n, ok in self._ok.items() if ok]

    def blocked(self, limits):
        """Return (sensor, cm) for the first sensor closer than its limit, else None.
        limits: {sensor_name: cm}"""
        d = self.distances()
        for name, lim in limits.items():
            v = d.get(name)
            if v is not None and v < lim:
                return name, v
        return None

    @staticmethod
    def label(name):
        """Direction a sensor faces, in words."""
        ang = config._angle(name)
        if ang == 0:
            return "straight ahead"
        if abs(ang) >= 135:
            return "behind"
        return f"front {'right' if ang > 0 else 'left'} ({abs(ang)} degrees off center)"

    def summary(self):
        """One short line for the LLM's context, e.g. "ahead 85 cm, front left clear, behind 40 cm"."""
        d = self.distances()
        dead = set(d) - set(self.working())
        parts = []
        for name, v in d.items():
            where = self.label(name).split(" (")[0]
            parts.append(f"{where} {'no reading' if name in dead else (f'{v:.0f} cm' if v is not None else 'clear')}")
        return ", ".join(parts)

    def describe(self):
        """Full reading for the read_sensors tool."""
        d = self.distances()
        working = set(self.working())
        out = {"readings": {}, "stop_distance_cm": config.OBSTACLE_STOP_CM}
        for name, v in d.items():
            out["readings"][self.label(name)] = (
                "no reading (sensor may be disconnected)" if name not in working
                else f"{v:.0f} cm" if v is not None else "clear (nothing within 3 m)")
        for direction, guard in (("forward", config.STOP_CM_FORWARD), ("backward", config.STOP_CM_BACKWARD)):
            hit = self.blocked(guard)
            if hit:
                out[f"can_move_{direction}"] = f"no, obstacle {hit[1]:.0f} cm away"
            else:
                near = [d[n] for n in guard if d.get(n) is not None]
                room = (min(near) - config.OBSTACLE_STOP_CM) / 100 if near else None
                out[f"can_move_{direction}"] = f"yes, about {room:.1f} m of room" if room is not None else "yes, clear"
        return out

    def min_front(self):
        d = [v for k, v in self.distances().items() if k.startswith("front") and v is not None]
        return min(d) if d else None

    def rear(self):
        return self.distances().get("rear")

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        for s in self.sensors:
            s.cb.cancel()
            for pin in (s.trig, s.echo):
                try:
                    lgpio.gpio_free(self.h, pin)
                except lgpio.error:
                    pass
        lgpio.gpiochip_close(self.h)


def main():
    """Live readout: python3 -m rover.sensors"""
    arr = UltrasonicArray().start()
    try:
        while True:
            d = arr.distances()
            line = "  ".join(f"{k:>12}: {('%5.1f cm' % v) if v else '   --   '}" for k, v in d.items())
            print("\r" + line, end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        arr.close()


if __name__ == "__main__":
    main()
