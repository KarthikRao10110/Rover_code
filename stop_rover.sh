#!/usr/bin/env bash
# Stop the rover agent and its language-model servers.
pkill -TERM -f "python -m rover.main" && echo "rover stopped" || echo "rover was not running"
sleep 2
pkill -TERM -f "llama-server .*--port 808[0-9]" && echo "llm server stopped"
exit 0
