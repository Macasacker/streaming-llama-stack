#!/bin/sh
# DSV4.1 single engine launcher. Options: --config, --model, --webui, --vision, --dry-run, --stop.
set -eu
exec "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/start-dsv41-single.sh" --preset 262k --keep 2541 "$@"
