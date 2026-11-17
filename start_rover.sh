#!/usr/bin/env bash
# One command to start the whole rover: language models, voice, ears, face, wheels.
#   ./start_rover.sh              fullscreen face on the HDMI screen
#   ./start_rover.sh --windowed   face in a window
#   ./start_rover.sh --help       all options
cd "$(dirname "$(readlink -f "$0")")"
mkdir -p logs
export LG_WD="$PWD/logs"   # lgpio puts its temp pipe files here
# allow launching from SSH / Claude: attach to the desktop session's display + audio
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
export DISPLAY="${DISPLAY:-:0}"
# only one rover at a time
exec 9>logs/.rover.lock
if ! flock -n 9; then
  echo "Rover is already running (stop it with ./stop_rover.sh)"; exit 1
fi
exec .venv/bin/python -m rover.main "$@" 2>&1 | tee -a logs/rover.log
