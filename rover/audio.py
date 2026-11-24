"""Keep the Bluetooth speaker/mic connected, and switch it between music and call mode.

A Bluetooth speaker can't play in music mode (A2DP: stereo, loud, clear) while its mic is in
use: the mic only works in call mode (HFP: 8-16 kHz, quieter). WirePlumber switches to call mode
by itself ~0.5 s after a recording opens. So the rover closes the mic while it talks and calls
music_mode() before the first word; reopening the mic afterwards switches back to call mode.
"""

import json
import re
import subprocess
import threading
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


_mode_lock = threading.Lock()


def _card():
    return "bluez_card." + config.BT_AUDIO_MAC.replace(":", "_")


def _pactl_json(*args):
    out = subprocess.run(["pactl", "-f", "json", *args], capture_output=True, text=True, timeout=5).stdout
    return json.loads(out or "[]")


def _music_ready(card):
    """True when the card is in an A2DP profile and its music-quality sink exists."""
    cards = [c for c in _pactl_json("list", "cards") if c.get("name") == card]
    if not cards or not cards[0].get("active_profile", "").startswith("a2dp-sink"):
        return False, cards[0] if cards else None
    prefix = "bluez_output." + config.BT_AUDIO_MAC.replace(":", "_")
    for sink in _pactl_json("list", "sinks", "short"):
        m = re.search(r"(\d+)Hz", sink.get("sample_specification", ""))
        if sink.get("name", "").startswith(prefix) and m and int(m.group(1)) >= 32000:  # call mode: 8/16 kHz
            return True, cards[0]
    return False, cards[0]


def music_mode(timeout=4.0, log=print):
    """Switch the Bluetooth speaker to music mode (A2DP) and wait until it's ready.
    Call it only while the mic is closed. Safe to call repeatedly; returns True when ready."""
    if not (config.BT_AUDIO_MAC and getattr(config, "BT_MUSIC_WHILE_SPEAKING", False)):
        return False
    with _mode_lock:
        try:
            card = _card()
            ready, info = _music_ready(card)
            if ready:
                return True
            if not info:
                return False                      # speaker not connected
            profiles = info.get("profiles", {})
            name = "a2dp-sink" if profiles.get("a2dp-sink", {}).get("available", True) else \
                next((p for p in profiles if p.startswith("a2dp-sink")), None)
            if not name:
                return False
            subprocess.run(["pactl", "set-card-profile", card, name], capture_output=True, timeout=5)
            end = time.monotonic() + timeout
            while time.monotonic() < end:
                if _music_ready(card)[0]:
                    time.sleep(0.1)               # let the default sink settle on the new node
                    return True
                time.sleep(0.05)
            log("[audio] speaker didn't switch to music mode in time; speaking in call mode")
        except Exception as e:
            log(f"[audio] music mode switch failed: {e}")
        return False
