#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    echo '请在本机桌面终端运行此脚本，以显示实时调试窗口。' >&2
    exit 1
fi
exec ./tools/run_simulator.sh --display "$@"
