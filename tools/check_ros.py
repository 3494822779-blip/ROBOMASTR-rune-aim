#!/usr/bin/env python3
"""Integration check against a running virtual demo / Foxglove bridge."""
import json
import math
import time
import cv2
import numpy as np
import websocket
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Vector3Stamped
from diagnostic_msgs.msg import DiagnosticArray
from visualization_msgs.msg import MarkerArray

expected = {
    '/rune/image/compressed': CompressedImage,
    '/rune/aim': Vector3Stamped,
    '/rune/status': DiagnosticArray,
    '/rune/markers': MarkerArray,
}
ws = websocket.create_connection('ws://127.0.0.1:8765',
    subprotocols=['foxglove.sdk.v1', 'foxglove.websocket.v1'], timeout=5)
subscriptions = {}
seen = set()
deadline = time.monotonic() + 20
try:
    while time.monotonic() < deadline and seen != set(expected):
        msg = ws.recv()
        if isinstance(msg, str):
            event = json.loads(msg)
            if event.get('op') == 'advertise':
                for channel in event['channels']:
                    topic = channel['topic']
                    if topic in expected:
                        sid = channel['id']
                        subscriptions[sid] = topic
                        ws.send(json.dumps({'op': 'subscribe', 'subscriptions': [
                            {'id': sid, 'channelId': channel['id']}]}))
        elif msg and msg[0] == 1:
            sid = int.from_bytes(msg[1:5], 'little')
            topic = subscriptions[sid]
            data = deserialize_message(msg[13:], expected[topic])
            if topic.endswith('compressed'):
                image = cv2.imdecode(np.frombuffer(bytes(data.data), np.uint8), cv2.IMREAD_COLOR)
                assert image is not None and image.shape == (1080, 1440, 3)
                assert image.max() > 0, 'Expected visible detection/status drawing'
            elif topic.endswith('aim'):
                assert all(math.isfinite(v) for v in (data.vector.x, data.vector.y, data.vector.z))
                assert data.vector.z > 0
            elif topic.endswith('markers'):
                if len(data.markers) <= 1:
                    continue
                assert all(m.header.frame_id == 'odom' for m in data.markers)
            elif topic.endswith('status'):
                values = {v.key: v.value for v in data.status[0].values}
                if values['found'] != '1':
                    continue
            seen.add(topic)
            print('PASS', topic)
    assert seen == set(expected), f'Missing data: {set(expected) - seen}'
    print('PASS: Foxglove handshake, channel discovery, CDR decoding, image and telemetry')
finally:
    ws.close()
