"""Robot-to-robot messages over the local network (both rovers on the same Wi-Fi).

Each robot runs a tiny TCP server on LINK_PORT. A message is one JSON line in, one JSON line
out; every message carries the shared LINK_KEY. Used for robot-vs-robot debates and by
tools/debate.py to start one from the command line.

    link = PeerLink(on_message=handler).start()
    link.send({"type": "turn", "text": "..."})        # -> reply dict or None if unreachable

Finding the other robot: PEER_HOST is tried first (best as a name like "raspberrypi.local", so
a changed IP doesn't matter). If that fails, the local network is scanned for a robot that
answers on the link port with our LINK_KEY, and the address found is remembered.
"""

import ipaddress
import json
import socket
import socketserver
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import config

DISCOVER_EVERY_S = 20        # don't rescan the network more often than this
SCAN_TIMEOUT_S = 0.6         # per address while scanning


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            msg = json.loads(self.rfile.readline(65536).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return
        if not isinstance(msg, dict) or msg.get("key") != config.LINK_KEY:
            reply = {"ok": False, "error": "bad key"}
        elif msg.get("type") == "ping":             # discovery: "who are you?"
            reply = {"ok": True, "name": config.ROBOT_NAME}
        else:
            msg.pop("key", None)
            try:
                reply = self.server.on_message(msg) or {"ok": True}
            except Exception as e:  # never let a bad message kill the server
                reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        self.wfile.write((json.dumps(reply) + "\n").encode("utf-8"))


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _exchange(host, port, data, timeout):
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.sendall(data)
        s.settimeout(timeout)
        line = s.makefile("rb").readline(65536)
    return json.loads(line.decode("utf-8")) if line else {"ok": True}


def _my_ip():
    """This Pi's LAN address (no packet is actually sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


class PeerLink:
    def __init__(self, on_message, log=print):
        self.on_message = on_message
        self.log = log
        self.server = None
        self.host = config.PEER_HOST          # where the other robot is (updated by discovery)
        self._last_scan = 0.0
        self._scan_lock = threading.Lock()

    def start(self):
        self.server = _Server(("0.0.0.0", config.LINK_PORT), _Handler)
        self.server.on_message = self.on_message
        threading.Thread(target=self.server.serve_forever, name="link", daemon=True).start()
        return self

    @staticmethod
    def configured():
        return bool(config.PEER_HOST and config.PEER_NAME)

    def send(self, msg, host=None, tries=3, timeout=5.0):
        """Send to the other robot. Returns its reply dict, or None if it can't be reached."""
        explicit = host is not None
        port = config.LINK_PORT if explicit else (config.PEER_PORT or config.LINK_PORT)
        if not (host or self.host):
            return None
        data = (json.dumps(dict(msg, key=config.LINK_KEY)) + "\n").encode("utf-8")
        err = None
        for i in range(tries):
            target = host or self.host
            try:
                return _exchange(target, port, data, timeout)
            except (OSError, ValueError) as e:
                err = e
                if not explicit and isinstance(e, OSError):
                    scanned = time.monotonic() - self._last_scan >= DISCOVER_EVERY_S
                    found = self.discover(port)
                    if found and found != target:
                        continue                   # retry right away at the new address
                    if scanned and not found:
                        break                      # just searched the whole network: it isn't there
                if i < tries - 1:
                    time.sleep(1.0)
        self.log(f"[link] can't reach {config.PEER_NAME or 'the other robot'} at {host or self.host}:{port}: {err}")
        return None

    # ---- finding the other robot ------------------------------------------------------------
    def _is_peer(self, host, port):
        """True if a robot with our LINK_KEY (other than me) answers at host:port."""
        try:
            r = _exchange(host, port, (json.dumps({"type": "ping", "key": config.LINK_KEY}) + "\n").encode(),
                          SCAN_TIMEOUT_S if host[0].isdigit() else 3.0)
        except (OSError, ValueError):
            return False
        if not isinstance(r, dict) or r.get("error") == "bad key":
            return False
        # robots running older code don't send their name; anything with our key that isn't me counts
        return r.get("name", "").lower() != config.ROBOT_NAME.lower()

    def discover(self, port=None):
        """Look for the other robot: first PEER_HOST (name lookup), then scan this Wi-Fi network.
        Returns its address (and remembers it), or None."""
        port = port or config.PEER_PORT or config.LINK_PORT
        with self._scan_lock:
            if time.monotonic() - self._last_scan < DISCOVER_EVERY_S:
                return None
            self._last_scan = time.monotonic()
            peer = config.PEER_NAME or "the other robot"
            if config.PEER_HOST and config.PEER_HOST != self.host and self._is_peer(config.PEER_HOST, port):
                self.host = config.PEER_HOST
                return self.host
            me = _my_ip()
            if not me:
                return None
            self.log(f"[link] looking for {peer} on the network...")
            net = ipaddress.ip_network(f"{me}/24", strict=False)
            candidates = [str(ip) for ip in net.hosts() if str(ip) != me]
            with ThreadPoolExecutor(max_workers=64) as ex:
                hits = [ip for ip, ok in zip(candidates, ex.map(lambda ip: self._is_peer(ip, port), candidates)) if ok]
            if not hits:
                self.log(f"[link] {peer} not found on {net} (is it switched on, running, and on this Wi-Fi?)")
                return None
            self.host = hits[0]
            self.log(f"[link] found {peer} at {self.host}")
            return self.host

    def check(self):
        """Is the other robot reachable RIGHT NOW? Tries the address we know first (generous
        timeout: weak Wi-Fi), then searches the network, ignoring the rescan cooldown.
        Returns (address, round-trip ms) or (None, None)."""
        port = config.PEER_PORT or config.LINK_PORT
        ping = (json.dumps({"type": "ping", "key": config.LINK_KEY}) + "\n").encode()
        for host in dict.fromkeys(h for h in (self.host, config.PEER_HOST) if h):
            t = time.monotonic()
            try:
                r = _exchange(host, port, ping, 3.0)
            except (OSError, ValueError):
                continue
            if isinstance(r, dict) and r.get("ok") and r.get("name", "").lower() != config.ROBOT_NAME.lower():
                self.host = host
                return host, int((time.monotonic() - t) * 1000)
        with self._scan_lock:
            self._last_scan = 0.0
        t = time.monotonic()
        host = self.discover(port)
        return (host, int((time.monotonic() - t) * 1000)) if host else (None, None)

    def close(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
