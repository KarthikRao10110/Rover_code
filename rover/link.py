"""Robot-to-robot messages over the local network (both rovers on the same Wi-Fi).

Each robot runs a tiny TCP server on LINK_PORT. A message is one JSON line in, one JSON line
out; every message carries the shared LINK_KEY. Used for robot-vs-robot debates and by
tools/debate.py to start one from the command line.

    link = PeerLink(on_message=handler).start()
    link.send({"type": "turn", "text": "..."})        # -> reply dict or None if unreachable
"""

import json
import socket
import socketserver
import threading
import time

from . import config


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            msg = json.loads(self.rfile.readline(65536).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return
        if not isinstance(msg, dict) or msg.get("key") != config.LINK_KEY:
            reply = {"ok": False, "error": "bad key"}
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


class PeerLink:
    def __init__(self, on_message, log=print):
        self.on_message = on_message
        self.log = log
        self.server = None

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
        port = config.LINK_PORT if host else (config.PEER_PORT or config.LINK_PORT)
        host = host or config.PEER_HOST
        if not host:
            return None
        data = (json.dumps(dict(msg, key=config.LINK_KEY)) + "\n").encode("utf-8")
        for i in range(tries):
            try:
                with socket.create_connection((host, port), timeout=timeout) as s:
                    s.sendall(data)
                    s.settimeout(timeout)
                    line = s.makefile("rb").readline(65536)
                return json.loads(line.decode("utf-8")) if line else {"ok": True}
            except (OSError, ValueError) as e:
                if i == tries - 1:
                    self.log(f"[link] can't reach {host}:{port}: {e}")
                    return None
                time.sleep(1.0)

    def close(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
