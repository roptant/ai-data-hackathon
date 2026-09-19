#!/bin/sh
# Builds the UI with the pinned TypeScript compiler; no npm install required.
# TSC may point at any typescript@5.9.3 checkout (integrity pinned in ui/package.json).
set -eu
cd "$(dirname "$0")/../ui"
TSC="${TSC:-$HOME/.local/opt/typescript-5.9.3/bin/tsc}"
if [ ! -f "$TSC" ]; then
  TSC="$(command -v tsc)"
fi
node "$TSC" -p tsconfig.json
node ../scripts/copy-ui-assets.mjs
