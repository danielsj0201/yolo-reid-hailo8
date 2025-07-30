import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
from collections import defaultdict

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_reid_pipeline import GStreamerDetectionReIDApp


class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42
        self.id_counter = defaultdict(int)
        self.frame_threshold = 300
        self.target_id = None
        self.prev_distance = None

    def new_function(self):
        return "The meaning of life is: "


def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    current_frame = user_data.get_count()
    string_to_print = f"Frame count: {current_frame}\n"

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    for detection in detections:
        if detection.get_label() != "person":
            continue

        bbox = detection.get_bbox()
        center_x = (bbox.xmin() + bbox.xmax()) / 2
        center_y = (bbox.ymin() + bbox.ymax()) / 2
        confidence = detection.get_confidence()

        track_id = 0
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if track:
            track_id = track[0].get_id()

        # Feature vector
        feature_vec_str = "No feature vector"
        matrices = roi.get_objects_typed(hailo.HAILO_MATRIX)
        if matrices:
            vec = matrices[0].get_matrix()
            feature_vec_str = ', '.join(f"{v:.4f}" for v in vec[:5])

        string_to_print += (
            f"Detection: ID: {track_id} Label: person Confidence: {confidence:.2f}\n"
            f"Center: ({center_x:.4f}, {center_y:.4f})\n"
            f"Feature Vector (first 5): {feature_vec_str}\n"
        )

    print(string_to_print)
    return Gst.PadProbeReturn.OK


if __name__ == "__main__":
    user_data = user_app_callback_class()
    app = GStreamerDetectionReIDApp(app_callback, user_data)
    app.run()
