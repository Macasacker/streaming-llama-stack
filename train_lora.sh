#!/bin/sh
# Thin wrapper over bin/slstack.
set -eu
exec "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/bin/slstack" --config ${cmd}.local.json train_lora "$@"
