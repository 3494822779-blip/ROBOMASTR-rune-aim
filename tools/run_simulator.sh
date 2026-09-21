#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ -z "${DAEDALUS_BRIDGE_TOKEN:-}" ]]; then
    source "${XDG_CONFIG_HOME:-$HOME/.config}/rune_aim/simulator.env"
fi
exec ./build/rune_aim -c config/rune_simulator.yaml "$@"
