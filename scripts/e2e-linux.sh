#!/bin/sh
# End-to-end check on a Linux desktop session (debug build):
# test audio -> live partials over the local API -> final transcript ->
# native AT-SPI insertion into a GTK entry -> read back via AT-SPI.
# Opens a zenity entry and the app briefly. Requires installed ASR model.
set -eu
cd "$(dirname "$0")/.."
WAV="${1:?usage: e2e-linux.sh <16kHz-mono.wav> [expected phrase]}"
EXPECT="${2:-}"
OUT="${E2E_OUT:-$(mktemp -d)}"
DATA="$HOME/.local/share/app.localdictation.desktop"
PORT=18765
TOKEN="ldc_e2e_$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')"
cargo run -q -p dictation-storage --example provision_api_client -- "$DATA/metadata/metadata.sqlite3" "e2e harness" "$TOKEN" \
  transcript:live transcript:final status:read session:control
printf '%s' "$TOKEN" > "$OUT/token"
LOCAL_DICTATION_DEBUG=1 \
LOCAL_DICTATION_WORKER_DIR="$PWD/target/release" \
LOCAL_DICTATION_TEST_AUDIO="$WAV" \
LOCAL_DICTATION_DEV_SETTINGS="{\"apiEnabled\":true,\"apiPort\":$PORT,\"automaticInsertion\":true,\"language\":\"en\"}" \
  ./target/debug/local-dictation-desktop > "$OUT/app.log" 2>&1 &
APP=$!
trap 'kill $APP 2>/dev/null || true; kill -9 ${EDITOR_PID:-0} 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
  if grep -q '"asrReady":true' "$OUT/app.log" && grep -q "\"apiListening\":$PORT" "$OUT/app.log"; then break; fi
  sleep 1
done
zenity --entry --title "Dictation test" --text "Test field" >/dev/null 2>&1 &
EDITOR_PID=$!
sleep 5
./target/debug/examples/caption_client captions --port $PORT --token-file "$OUT/token" --once > "$OUT/captions.log" 2>&1 &
CAPTIONS=$!
sleep 1
START=$(./target/debug/examples/caption_client start --port $PORT --token-file "$OUT/token")
echo "start: $START"
SESSION=$(printf '%s' "$START" | sed -n 's/.*"session_id":"\([^"]*\)".*/\1/p')
DURATION=$(( $(stat -c %s "$WAV") / 32000 + 2 ))
sleep 3
if [ -n "${E2E_PROBE_FOCUS:-}" ]; then
  ATSPI_DUMP=1 cargo run -q -p dictation-platform --example atspi_read_text -- zenity 2>&1 | grep "text box" > "$OUT/focus-during.txt" || true
fi
sleep $(( DURATION - 3 ))
./target/debug/examples/caption_client stop --port $PORT --token-file "$OUT/token" --session "$SESSION"
T0=$(date +%s%N)
wait $CAPTIONS || true
T1=$(date +%s%N)
echo "final arrived $(( (T1 - T0) / 1000000 )) ms after stop"
sleep 2
INSERTED=$(cargo run -q -p dictation-platform --example atspi_read_text -- zenity 2>&1 || true)
echo "--- captions"; cat "$OUT/captions.log"
echo "--- inserted into test field"; echo "$INSERTED"
grep -o '"notice":"[^"]*"' "$OUT/app.log" | tail -3
if [ -n "$EXPECT" ]; then
  if printf '%s' "$INSERTED" | grep -qi "$EXPECT"; then echo "PASS"; else echo "FAIL"; exit 1; fi
fi
