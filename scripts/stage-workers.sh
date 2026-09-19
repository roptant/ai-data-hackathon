#!/bin/sh
# Builds the two model workers and stages them as Tauri sidecars
# (src-tauri/binaries/<name>-<target-triple>[.exe]).
set -eu
cd "$(dirname "$0")/.."
TRIPLE="${TARGET_TRIPLE:-$(rustc -vV | sed -n 's/^host: //p')}"
case "$TRIPLE" in *windows*) EXE=.exe ;; *) EXE= ;; esac
cargo build --release --target "$TRIPLE" -p dictation-asr-worker -p dictation-privacy-worker
mkdir -p src-tauri/binaries
for worker in dictation-asr-worker dictation-privacy-worker; do
  cp "target/$TRIPLE/release/$worker$EXE" "src-tauri/binaries/$worker-$TRIPLE$EXE"
done
echo "staged workers for $TRIPLE"
