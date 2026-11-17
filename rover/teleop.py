"""Low-latency manual control from a keyboard or gamepad via Linux evdev.

Works with USB / 2.4 GHz dongle / Bluetooth keyboards and gamepads. Devices
can be plugged in or unplugged while running.

Keyboard:
    W/S or Up/Down     forward / back        A/D or Left/Right   turn
    Shift (hold)       boost                 [ / ]               speed limit -/+
    1..9, 0            set speed limit 10%..90%, 100%
    Space              EMERGENCY STOP (latched)                  R  release e-stop
    Esc                quit
Gamepad:
    Left stick (or D-pad)  drive              RB / R1 (hold)      boost
    A / Cross              EMERGENCY STOP     B / Circle          release e-stop
    Start                  quit
"""

import argparse
import select
import signal
import sys
import time

import evdev
from evdev import ecodes as e

from . import config
from .drive import Drivetrain, use_mock_pins

FWD_KEYS = {e.KEY_W, e.KEY_UP}
BACK_KEYS = {e.KEY_S, e.KEY_DOWN}
LEFT_KEYS = {e.KEY_A, e.KEY_LEFT}
RIGHT_KEYS = {e.KEY_D, e.KEY_RIGHT}
BOOST_KEYS = {e.KEY_LEFTSHIFT, e.KEY_RIGHTSHIFT}
# 1..9 -> 10%..90%, 0 -> 100%
SPEED_KEYS = {getattr(e, f"KEY_{n}"): (n or 10) / 10 for n in range(10)}


def is_keyboard(dev):
    keys = dev.capabilities().get(e.EV_KEY, [])
    return e.KEY_W in keys and e.KEY_SPACE in keys


def is_gamepad(dev):
    caps = dev.capabilities()
    keys = caps.get(e.EV_KEY, [])
    axes = [a[0] if isinstance(a, tuple) else a for a in caps.get(e.EV_ABS, [])]
    return (e.BTN_SOUTH in keys or e.BTN_GAMEPAD in keys) and e.ABS_X in axes


class Keyboard:
    def __init__(self, dev):
        self.dev = dev
        self.held = set()
        self.last_event = time.monotonic()
        self.has_repeat = e.EV_REP in dev.capabilities()

    def handle(self, ev, state):
        if ev.type != e.EV_KEY:
            return
        self.last_event = time.monotonic()
        if ev.value == 0:
            self.held.discard(ev.code)
            return
        if ev.value == 1:  # fresh press (2 = autorepeat)
            self.held.add(ev.code)
            if ev.code == e.KEY_SPACE:
                state.estop = True
            elif ev.code == e.KEY_R:
                state.estop = False
            elif ev.code == e.KEY_ESC:
                state.quit = True
            elif ev.code == e.KEY_LEFTBRACE:
                state.speed = max(0.1, round(state.speed - 0.1, 1))
            elif ev.code == e.KEY_RIGHTBRACE:
                state.speed = min(1.0, round(state.speed + 0.1, 1))
            elif ev.code in SPEED_KEYS:
                state.speed = SPEED_KEYS[ev.code]

    def command(self):
        """Return (throttle, turn, boost) or None if not driving."""
        # Wireless keyboards can drop a key-up when out of range. While a key is held the
        # kernel sends autorepeat events, so silence means the link is gone.
        if self.held and self.has_repeat and time.monotonic() - self.last_event > config.KEY_DEADMAN_SEC:
            self.held.clear()
        h = self.held
        throttle = (1 if h & FWD_KEYS else 0) - (1 if h & BACK_KEYS else 0)
        turn = (1 if h & RIGHT_KEYS else 0) - (1 if h & LEFT_KEYS else 0)
        if not throttle and not turn:
            return None
        return float(throttle), float(turn), bool(h & BOOST_KEYS)


class Gamepad:
    def __init__(self, dev):
        self.dev = dev
        self.axes = {}
        self.info = {code: info for code, info in dev.capabilities().get(e.EV_ABS, [])}
        self.boost = False

    def _norm(self, code):
        info = self.info.get(code)
        if info is None or code not in self.axes:
            return 0.0
        if code in (e.ABS_HAT0X, e.ABS_HAT0Y):
            return float(self.axes[code])
        mid = (info.max + info.min) / 2
        half = (info.max - info.min) / 2 or 1
        v = (self.axes[code] - mid) / half
        if abs(v) < config.STICK_DEADZONE:
            return 0.0
        # rescale so output starts at 0 just outside the deadzone
        sign = 1 if v > 0 else -1
        return sign * min(1.0, (abs(v) - config.STICK_DEADZONE) / (1 - config.STICK_DEADZONE))

    def handle(self, ev, state):
        if ev.type == e.EV_ABS:
            self.axes[ev.code] = ev.value
        elif ev.type == e.EV_KEY:
            if ev.code == e.BTN_TR:
                self.boost = ev.value != 0
            elif ev.value == 1:
                if ev.code == e.BTN_SOUTH:
                    state.estop = True
                elif ev.code == e.BTN_EAST:
                    state.estop = False
                elif ev.code == e.BTN_START:
                    state.quit = True

    def command(self):
        throttle = -self._norm(e.ABS_Y) or -self._norm(e.ABS_HAT0Y)
        turn = self._norm(e.ABS_X) or self._norm(e.ABS_HAT0X)
        if not throttle and not turn:
            return None
        return throttle, turn, self.boost


class State:
    def __init__(self):
        self.estop = False
        self.quit = False
        self.speed = config.DEFAULT_SPEED


def mix(throttle, turn, limit):
    """Arcade (one stick) -> tank. Spins in place when throttle is zero."""
    if abs(throttle) < 1e-3:
        left, right = turn * config.TURN_GAIN_SPIN, -turn * config.TURN_GAIN_SPIN
    else:
        k = config.TURN_GAIN_MOVING
        left, right = throttle + turn * k * abs(throttle), throttle - turn * k * abs(throttle)
        m = max(abs(left), abs(right), 1.0)
        left, right = left / m, right / m
    return left * limit, right * limit


class Controllers:
    def __init__(self, grab):
        self.grab = grab
        self.by_fd = {}
        self.last_scan = 0.0

    def scan(self):
        self.last_scan = time.monotonic()
        known = {c.dev.path for c in self.by_fd.values()}
        for path in evdev.list_devices():
            if path in known:
                continue
            try:
                dev = evdev.InputDevice(path)
                ctl = Gamepad(dev) if is_gamepad(dev) else Keyboard(dev) if is_keyboard(dev) else None
                if ctl is None:
                    dev.close()
                    continue
                if self.grab:
                    dev.grab()  # keystrokes no longer reach the desktop/terminal
                self.by_fd[dev.fd] = ctl
                log(f"connected: {type(ctl).__name__} '{dev.name}' ({path})")
            except OSError:
                pass

    def drop(self, fd):
        ctl = self.by_fd.pop(fd, None)
        if ctl:
            log(f"DISCONNECTED: '{ctl.dev.name}' -> stopping")
            try:
                ctl.dev.close()
            except OSError:
                pass

    def close(self):
        for fd in list(self.by_fd):
            ctl = self.by_fd.pop(fd)
            try:
                if self.grab:
                    ctl.dev.ungrab()
                ctl.dev.close()
            except OSError:
                pass


def log(msg):
    sys.stdout.write("\r\033[K" + msg + "\n")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="mock GPIO, motors do not move")
    ap.add_argument("--no-grab", action="store_true", help="don't grab the keyboard exclusively")
    args = ap.parse_args()

    if args.dry_run:
        use_mock_pins()

    def on_term(*_):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, on_term)

    drive = Drivetrain()
    ctls = Controllers(grab=not args.no_grab)
    state = State()
    period = 1.0 / config.CONTROL_HZ
    last = time.monotonic()
    last_print = 0.0
    log(__doc__.strip().split("\n\n", 1)[1])
    log("dry run: GPIO mocked" if args.dry_run else "LIVE: motors armed")

    try:
        ctls.scan()
        while not state.quit:
            if time.monotonic() - ctls.last_scan > 2.0:
                ctls.scan()

            fds = list(ctls.by_fd)
            ready = select.select(fds, [], [], period)[0] if fds else (time.sleep(period) or [])
            for fd in ready:
                ctl = ctls.by_fd.get(fd)
                try:
                    for ev in ctl.dev.read():
                        ctl.handle(ev, state)
                except OSError:
                    ctls.drop(fd)
                    drive.stop()

            # First controller with active input wins.
            cmd = None
            for ctl in ctls.by_fd.values():
                cmd = ctl.command()
                if cmd:
                    break

            if state.estop:
                drive.stop()
                left = right = 0.0
            elif cmd:
                throttle, turn, boost = cmd
                left, right = mix(throttle, turn, config.BOOST_SPEED if boost else state.speed)
                drive.set_target(left, right)
            else:
                left = right = 0.0
                drive.set_target(0.0, 0.0)

            now = time.monotonic()
            drive.update(now - last)
            last = now

            if now - last_print > 0.1:
                last_print = now
                tag = "E-STOP (R / B to release)" if state.estop else f"limit {state.speed:.1f}"
                sys.stdout.write(
                    f"\r\033[K[{tag}]  L {drive.left.current:+.2f}  R {drive.right.current:+.2f}"
                    f"  controllers: {len(ctls.by_fd)}"
                )
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        drive.close()
        ctls.close()
        log("stopped.")


if __name__ == "__main__":
    main()
