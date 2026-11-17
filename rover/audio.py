"""Keep the Bluetooth speaker/mic connected."""

import subprocess
import time

from . import config


def bt_connected(mac=None):
    mac = mac or config.BT_AUDIO_MAC
    if not mac:
        return True
    try:
        out = subprocess.run(["bluetoothctl", "info", mac], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return False
    return "Connected: yes" in out


def bt_mic_present(mac=None):
    mac = (mac or config.BT_AUDIO_MAC or "").replace(":", "_")
    try:
        out = subprocess.run(["pw-cli", "ls", "Node"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return False
    return f"bluez_input.{config.BT_AUDIO_MAC}" in out or f"bluez_input.{mac}" in out


def ensure_bt(log=print, timeout=20):
    """Connect the Bluetooth audio device if needed. Returns True when its mic is available."""
    if not config.BT_AUDIO_MAC:
        return True
    if bt_connected() and bt_mic_present():
        return True
    log(f"[audio] Bluetooth audio not connected, connecting {config.BT_AUDIO_MAC}...")
    try:
        subprocess.run(["bluetoothctl", "connect", config.BT_AUDIO_MAC],
                       capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    end = time.monotonic() + 8
    while time.monotonic() < end:  # the PipeWire nodes appear a moment after connecting
        if bt_connected() and bt_mic_present():
            log("[audio] Bluetooth audio connected")
            return True
        time.sleep(0.5)
    log("[audio] could not connect Bluetooth audio (is the speaker on and in range?)")
    return False


def mic_target():
    """PipeWire source to record from: the BT mic if configured, else the default."""
    return f"bluez_input.{config.BT_AUDIO_MAC}" if config.BT_AUDIO_MAC else None
