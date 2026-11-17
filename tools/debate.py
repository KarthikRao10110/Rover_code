"""Start or stop a robot-vs-robot debate from the command line (more reliable than saying a
long topic out loud). Run on either rover while ./start_rover.sh is running; that rover hosts.

    .venv/bin/python -m tools.debate "Homework should be banned"
    .venv/bin/python -m tools.debate "Is social media good for teenagers?" --side against
    .venv/bin/python -m tools.debate --stop
"""

import argparse

from rover import config
from rover.link import PeerLink


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("topic", nargs="?", help="statement or yes/no question")
    ap.add_argument("--side", choices=["for", "against"], default="for", help="this robot's side (default: for)")
    ap.add_argument("--stop", action="store_true", help="stop the debate that is running")
    args = ap.parse_args()
    if not args.stop and not args.topic:
        ap.error("give a topic, or --stop")
    msg = {"type": "stop"} if args.stop else {"type": "host", "topic": args.topic, "side": args.side}
    reply = PeerLink(on_message=None).send(msg, host="127.0.0.1", tries=1, timeout=30)
    if reply is None:
        print(f"The rover isn't running (nothing on port {config.LINK_PORT}). Start it with ./start_rover.sh")
    else:
        print(reply.get("message") or reply)


if __name__ == "__main__":
    main()
