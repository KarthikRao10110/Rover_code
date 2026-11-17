"""Skid-steer drivetrain on an L298N."""

from gpiozero import Device, DigitalOutputDevice, PWMOutputDevice

from . import config


def use_mock_pins():
    """Swap in gpiozero's mock pin factory so nothing physical moves."""
    from gpiozero.pins.mock import MockFactory, MockPWMPin

    Device.pin_factory = MockFactory(pin_class=MockPWMPin)


def _clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


class Side:
    """One L298N channel: IN pins set direction, EN pin carries PWM speed."""

    def __init__(self, fwd, back, enable, inverted):
        if inverted:
            fwd, back = back, fwd
        self.fwd = DigitalOutputDevice(fwd)
        self.back = DigitalOutputDevice(back)
        self.en = PWMOutputDevice(enable, frequency=config.PWM_FREQUENCY)
        self.current = 0.0  # last applied command (-1..1, before MIN_DUTY mapping)

    def raw(self, duty):
        """duty in -1..1 sent straight to the driver (no MIN_DUTY mapping)."""
        if abs(duty) < 1e-3:
            self.en.value = 0
            self.fwd.off()
            self.back.off()  # both low + EN low = coast
            return
        self.fwd.value = duty > 0
        self.back.value = duty < 0
        self.en.value = min(1.0, abs(duty))

    def apply(self, value):
        self.current = value
        if abs(value) < 1e-3:
            self.raw(0)
        else:
            sign = 1 if value > 0 else -1
            self.raw(sign * (config.MIN_DUTY + (1.0 - config.MIN_DUTY) * abs(value)))

    def close(self):
        self.raw(0)
        for d in (self.en, self.fwd, self.back):
            d.close()


class Drivetrain:
    def __init__(self):
        self.left = Side(config.LEFT_IN1, config.LEFT_IN2, config.LEFT_EN, config.LEFT_INVERTED)
        self.right = Side(config.RIGHT_IN3, config.RIGHT_IN4, config.RIGHT_EN, config.RIGHT_INVERTED)
        self.target = (0.0, 0.0)

    def set_target(self, left, right):
        self.target = (_clamp(left), _clamp(right))

    def update(self, dt):
        """Move actual motor commands toward target, limited by ACCEL_PER_SEC."""
        step = config.ACCEL_PER_SEC * dt
        for side, tgt in ((self.left, self.target[0]), (self.right, self.target[1])):
            cur = side.current
            new = cur + _clamp(tgt - cur, -step, step)
            if new != cur:
                side.apply(new)

    def stop(self):
        """Immediate stop, no ramp."""
        self.target = (0.0, 0.0)
        self.left.apply(0.0)
        self.right.apply(0.0)

    def close(self):
        self.stop()
        self.left.close()
        self.right.close()
