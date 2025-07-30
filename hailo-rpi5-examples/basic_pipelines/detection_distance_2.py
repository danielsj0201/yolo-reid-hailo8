import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import socket
from collections import defaultdict

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42
        self.prev_distance = None

        # UDP socket configuration
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.fpga_ip = "192.168.100.200"
        self.port = 5005

        # ID tracking logic
        self.id_counter = defaultdict(int)
        self.frame_threshold = 300
        self.target_id = None

    def new_function(self):
        return "The meaning of life is: "

# -----------------------------------------------------------------------------------------------
# Callback function
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    current_frame = user_data.get_count()
    string_to_print = f"Frame count: {current_frame}\n"

    format, width, height = get_caps_from_pad(pad)

    frame = None
    if user_data.use_frame and format and width and height:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    detection_count = 0
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()

        if label == "person":
            track_id = 0
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = track[0].get_id()

            # ID 학습 단계
            if current_frame <= user_data.frame_threshold:
                user_data.id_counter[track_id] += 1
                if current_frame == user_data.frame_threshold:
                    # 가장 많이 등장한 ID 선택
                    user_data.target_id = max(user_data.id_counter, key=user_data.id_counter.get)
                    print(f"Target ID selected: {user_data.target_id}")
                continue  # 학습 중일 땐 전송 안 함

            # 이후는 target_id만 처리
            if user_data.target_id is not None and track_id != user_data.target_id:
                continue

            # 거리 계산
            xmin = bbox.xmin()
            ymin = bbox.ymin()
            xmax = bbox.xmax()
            ymax = bbox.ymax()
            width = bbox.width()
            height = bbox.height()
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2

            ref_width = 0.1200
            ref_height = 0.5986
            ref_distance = 300.0

            standard = ref_width / ref_height
            current_ratio = width / height
            if standard <= current_ratio:
                distance = ref_distance * (ref_width / width)
            else:
                distance = ref_distance * (ref_height / height)

            if user_data.prev_distance is not None:
                if abs(distance - user_data.prev_distance) > 100.0:
                    distance = user_data.prev_distance
            user_data.prev_distance = distance

            # 전송
            message = f"{center_x:.4f},{center_y:.4f},{distance:.2f}"
            user_data.sock.sendto(message.encode(), (user_data.fpga_ip, user_data.port))
            print(f"Sent message: {message}")

            string_to_print += (
                f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f} "
                f"Center:({center_x:.4f}, {center_y:.4f})\n"
                f"Distance: {distance:.2f}"
            )
            detection_count += 1

    if user_data.use_frame and frame is not None:
        cv2.putText(frame, f"Detections: {detection_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"{user_data.new_function()} {user_data.new_variable}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    print(string_to_print)
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
