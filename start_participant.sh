#!/bin/bash
# Start AgentX participant services: Arena (port 8000) + Solver (port 8001)
# Usage: bash start_participant.sh [host]

HOST=${1:-127.0.0.1}
BASE="$(cd "$(dirname "$0")" && pwd)"

echo "Starting AgentX participant services (host=$HOST)..."

# Start solver in background
cd "$BASE/src/solver" && python server.py --host "$HOST" --port 8001 &
echo "  solver → port 8001 (PID $!)"

sleep 2

# Start arena in foreground (keeps the process alive)
echo "  arena → port 8000"
cd "$BASE/src/arena" && exec python server.py --host "$HOST" --port 8000
