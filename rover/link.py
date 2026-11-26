"""Robot-to-robot messages over Bluetooth (RFCOMM), directly Pi to Pi.

Each robot listens on Bluetooth RFCOMM channel LINK_BT_CHANNEL. A message is one JSON line in,
one JSON line out; every message carries the shared LINK_KEY. Used for robot-vs-robot debates.
The other robot is found by its Bluetooth address (PEER_BT_MAC); pair the two Pis once.

The connection stays open all the time: a background thread connects at start, pings the other
robot every LINK_KEEPALIVE_S, and reconnects by itself when the link drops (other robot switched
off, out of range...). Each robot keeps its own outgoing connection to the other one.

    link = PeerLink(on_message=handler).start()
    link.send({"type": "turn", "text": "..."})        # -> reply dict or None if unreachable

The same handler also listens on a local Unix socket (LOCAL_SOCKET), so tools/debate.py on this
robot can start or stop a debate from the command line.
"""

import json
import os
import socket
import socketserver
import threading
import time

from . import config

LINK_CONNECT_TIMEOUT_S = 8.0    # Bluetooth connect attempt
LINK_PING_TIMEOUT_S = 6.0       # keep-alive / link check reply
LINK_RETRY_MAX_S = 30.0         # while the other robot is away, retry at most this far apart

LOCAL_SOCKET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "link.sock")


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        # many messages per connection: the other robot keeps its connection open
        while True:
            try:
                line = self.rfile.readline(65536)
            except OSError:
                return
            if not line:
                return                                   # the other side closed the connection
            try:
                msg = json.loads(line.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return
            try:
                self.wfile.write((json.dumps(self._reply(msg)) + "\n").encode("utf-8"))
                self.wfile.flush()
            except OSError:
                return

    def _reply(self, msg):
        if not isinstance(msg, dict) or msg.get("key") != config.LINK_KEY:
            reply = {"ok": False, "error": "bad key"}
        elif msg.get("type") == "ping":             # link check: "who are you?"
            reply = {"ok": True, "name": config.ROBOT_NAME}
        else:
            msg.pop("key", None)
            try:
                reply = self.server.on_message(msg) or {"ok": True}
            except Exception as e:  # never let a bad message kill the server
                reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        return reply


class _BtServer(socketserver.ThreadingTCPServer):
    """JSON-line server on a Bluetooth RFCOMM channel."""
    address_family = socket.AF_BLUETOOTH
    allow_reuse_address = False
    daemon_threads = True

    def __init__(self, channel, handler):
        super().__init__(("00:00:00:00:00:00", channel), handler, bind_and_activate=False)
        self.socket.close()             # base class made an L2CAP socket; we want RFCOMM
        self.socket = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
        try:
            self.server_bind()
            self.server_activate()
        except OSError:
            self.server_close()
            raise


class _LocalServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


def _read_reply(s):
    line = s.makefile("rb").readline(65536)
    return json.loads(line.decode("utf-8")) if line else {"ok": True}


def _encode(msg):
    return (json.dumps(dict(msg, key=config.LINK_KEY)) + "\n").encode("utf-8")


def send_local(msg, timeout=30):
    """Send a message to the rover app running on THIS Pi (used by tools/debate.py).
    Returns its reply dict, or None if the app isn't running."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(LOCAL_SOCKET)
            s.sendall(_encode(msg))
            return _read_reply(s)
    except (OSError, ValueError):
        return None


class PeerLink:
    def __init__(self, on_message, log=print):
        self.on_message = on_message
        self.log = log
        self.bt_server = None
        self.local_server = None
        self._sock = None                 # the open connection to the other robot (None = down)
        self._rfile = None
        self._lock = threading.Lock()     # one request/reply at a time on the connection
        self._stop = threading.Event()
        self._wake = threading.Event()    # asks the keeper thread to reconnect right now
        self._last_ok = 0.0
        self._was_connected = None

    def start(self):
        self.bt_server = _BtServer(config.LINK_BT_CHANNEL, _Handler)
        self.bt_server.on_message = self.on_message
        threading.Thread(target=self.bt_server.serve_forever, name="link-bt", daemon=True).start()
        self.log(f"[link] listening on Bluetooth RFCOMM channel {config.LINK_BT_CHANNEL}")
        try:
            os.makedirs(os.path.dirname(LOCAL_SOCKET), exist_ok=True)
            if os.path.exists(LOCAL_SOCKET):
                os.unlink(LOCAL_SOCKET)          # left over from a run that didn't close cleanly
            self.local_server = _LocalServer(LOCAL_SOCKET, _Handler)
            self.local_server.on_message = self.on_message
            threading.Thread(target=self.local_server.serve_forever, name="link-local", daemon=True).start()
        except OSError as e:
            self.log(f"[link] command-line control (tools/debate.py) disabled: {e}")
        if self.configured():
            threading.Thread(target=self._keep_connected, name="link-keeper", daemon=True).start()
        return self

    @staticmethod
    def configured():
        return bool(config.PEER_NAME and config.PEER_BT_MAC)

    @property
    def connected(self):
        return self._sock is not None

    # ---- the always-open connection ---------------------------------------------------------
    def _connect(self):
        """Open the connection (caller holds self._lock). Returns True when connected."""
        if self._sock is not None:
            return True
        s = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
        try:
            s.settimeout(LINK_CONNECT_TIMEOUT_S)
            s.connect((config.PEER_BT_MAC, config.LINK_BT_CHANNEL))
        except OSError as e:
            s.close()
            self._note(False, e)
            return False
        self._sock, self._rfile = s, s.makefile("rb")
        return True

    def _drop(self):
        """Close a broken connection (caller holds self._lock); the keeper reconnects."""
        if self._sock is not None:
            for f in (self._rfile, self._sock):
                try:
                    f.close()
                except OSError:
                    pass
            self._sock = self._rfile = None
            self._wake.set()

    def _request(self, data, timeout):
        """One message out, one reply in, on the open connection (caller holds self._lock).
        Any error or timeout closes the connection, so a late reply can't be mixed up with the
        next one."""
        if not self._connect():
            return None
        try:
            self._sock.settimeout(timeout)
            self._sock.sendall(data)
            line = self._rfile.readline(65536)
            if not line:
                raise OSError("connection closed by the other robot")
            reply = json.loads(line.decode("utf-8"))
        except (OSError, ValueError) as e:
            self._drop()
            self._note(False, e)
            return None
        self._last_ok = time.monotonic()
        self._note(True)
        return reply

    def _note(self, up, err=None):
        """Log only changes: connected / lost."""
        peer = config.PEER_NAME or "the other robot"
        if up and self._was_connected is not True:
            self.log(f"[link] connected to {peer} over Bluetooth ({config.PEER_BT_MAC})")
        elif not up and self._was_connected is not False:
            self.log(f"[link] no Bluetooth link to {peer} ({config.PEER_BT_MAC}): {err}; "
                     f"will keep trying")
        self._was_connected = up

    def _ping_data(self):
        return (json.dumps({"type": "ping", "key": config.LINK_KEY}) + "\n").encode()

    def _keep_connected(self):
        """Background: stay connected, ping when idle, reconnect with back-off when down."""
        backoff = 2.0
        while not self._stop.is_set():
            with self._lock:
                if time.monotonic() - self._last_ok >= config.LINK_KEEPALIVE_S or self._sock is None:
                    self._request(self._ping_data(), LINK_PING_TIMEOUT_S)
                up = self._sock is not None
            if up:
                backoff = 2.0
                wait = config.LINK_KEEPALIVE_S
            else:
                # don't page for the other robot too often: it shares the radio with the speaker
                wait, backoff = backoff, min(backoff * 2, LINK_RETRY_MAX_S)
            self._wake.clear()
            self._wake.wait(wait)

    # ---- API used by the debate code ------------------------------------------------------------
    def send(self, msg, tries=3, timeout=10.0):
        """Send to the other robot. Returns its reply dict, or None if it can't be reached."""
        if not self.configured():
            return None
        data = _encode(msg)
        for i in range(tries):
            with self._lock:
                reply = self._request(data, timeout)
            if reply is not None:
                return reply
            if i < tries - 1:
                time.sleep(1.0)
        self.log(f"[link] couldn't send '{msg.get('type')}' to {config.PEER_NAME or 'the other robot'}")
        return None

    def check(self):
        """Is the other robot reachable RIGHT NOW? Returns (its Bluetooth address, round-trip ms)
        or (None, None)."""
        if not self.configured():
            return None, None
        t = time.monotonic()
        with self._lock:
            r = self._request(self._ping_data(), LINK_PING_TIMEOUT_S)
        if isinstance(r, dict) and r.get("ok") and r.get("name", "").lower() != config.ROBOT_NAME.lower():
            return config.PEER_BT_MAC, int((time.monotonic() - t) * 1000)
        return None, None

    def close(self):
        self._stop.set()
        self._wake.set()
        with self._lock:
            self._drop()
        for srv in (self.bt_server, self.local_server):
            if srv:
                srv.shutdown()
                srv.server_close()
        if self.local_server:
            try:
                os.unlink(LOCAL_SOCKET)
            except OSError:
                pass
