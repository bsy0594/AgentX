#!/bin/bash
# Start AgentX participant services: Solver (port 8001, internal) + Arena (forwarded args)
# Usage: bash start_participant.sh [--host HOST] [--port PORT] [--card-url URL]

BASE="$(cd "$(dirname "$0")" && pwd)"
ARENA_ARGS="$@"

# Parse host from args for solver binding (defaults to 0.0.0.0)
HOST="0.0.0.0"
while [[ $# -gt 0 ]]; do
  case $1 in
    --host) HOST="$2"; shift 2;;
    *) shift;;
  esac
done

echo "Starting AgentX participant services..."

# Start solver in background on internal port 8001
cd "$BASE/src/solver" && python server.py --host "$HOST" --port 8001 &
echo "  solver → port 8001 (PID $!)"

sleep 2

# Start arena in foreground with all passed args
echo "  arena → $ARENA_ARGS"
cd "$BASE/src/arena" && exec python server.py $ARENA_ARGS
