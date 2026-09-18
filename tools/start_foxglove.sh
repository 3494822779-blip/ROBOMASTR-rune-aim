#!/usr/bin/env bash
# Starts both the optional ROS executable and Foxglove bridge. Ctrl+C stops both.
set -eo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source /opt/ros/humble/setup.bash
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/rune-ros-logs}"
if [[ ! -x build-ros/rune_aim ]]; then
  echo '先运行 tools/build_ros.sh' >&2
  exit 1
fi
# Do not silently reuse a bridge on an unknown ROS domain.
python3 - <<'PY'
import socket
s=socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(('0.0.0.0',8765))
except OSError:
    raise SystemExit('8765 端口已占用，请先停止原桥接程序，再运行本脚本。')
finally:
    s.close()
PY
bridge_pid=''
aim_pid=''
cleanup() {
  trap - EXIT INT TERM
  [[ -z "$aim_pid" ]] || kill -TERM "$aim_pid" 2>/dev/null || true
  [[ -z "$bridge_pid" ]] || kill -TERM "$bridge_pid" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
/opt/ros/humble/lib/foxglove_bridge/foxglove_bridge --ros-args -p address:=0.0.0.0 -p port:=8765 &
bridge_pid=$!
if [[ $# -eq 0 ]]; then set -- -c config/rune_gimbal_virtual.yaml; fi
./build-ros/rune_aim --ros --no-display "$@" &
aim_pid=$!
echo 'Foxglove: ws://机器人IP:8765；默认场景为虚拟符。Ctrl+C 停止。'
wait -n "$bridge_pid" "$aim_pid"
