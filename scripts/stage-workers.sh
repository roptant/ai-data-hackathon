#!/bin/sh
# POSIX convenience wrapper. The Node implementation is also usable directly
# from PowerShell and cmd.exe on Windows.
set -eu
exec node "$(dirname "$0")/stage-workers.mjs" "$@"
