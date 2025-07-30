import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import socket

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42  # New variable example
        self.prev_distance = None

        #UDP socket configuration
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.fpga_ip = "192.168.100.200"
        self.port = 5005
        
    def new_function(self):  # New function example
        return "The meaning of life is: "

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Using the user_data to count the number of frames
    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Parse the detections
    detection_count = 0
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        #if label == "person":
            # Get track ID
           # track_id = 0
           # track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
           # if len(track) == 1:
            #    track_id = track[0].get_id()
          #  xmin=bbox.xmin()
           # ymin=bbox.ymin()
          #  xmax=bbox.xmax()
           # ymax=bbox.ymax()
           # width=bbox.width()
           # height=bbox.height()
           # center_x=((xmin+xmax)/2)*width
           # center_y=((ymin+ymax)/2)*height
           # string_to_print += (f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}"
			#	f" Center:({center_x:.4f}, {center_y:.4f})\n")
           # detection_count += 1
        if label == "person":
            track_id = 0
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = track[0].get_id()
            xmin=bbox.xmin()
            ymin=bbox.ymin()
            xmax=bbox.xmax()
            ymax=bbox.ymax()
            width=bbox.width()
            height=bbox.height()
            center_x=((xmin+xmax)/2)
            center_y=((ymin+ymax)/2)
            ref_width = 0.1200 #측정 complete
            ref_height = 0.5986 #측정 complete
            ref_distance = 300.0 #cm
            standard = ref_width/ref_height
            current_ratio = width/height
            #print(f"{width:.4f}, {height:.4f}")
            if standard <= current_ratio:   #width 값의 오차율이 height보다 작을 때
                distance = ref_distance * (ref_width/width)
            else:   #height 값의 오차율이 width보다 작을 떄떄
                distance = ref_distance * (ref_height/height)

            if user_data.prev_distance is not None:
                if abs(distance - user_data.prev_distance) > 100.0:
                    distance = user_data.prev_distance
                
            user_data.prev_distance = distance

            message = f"{center_x:.4f},{center_y:.4f},{distance:.2f}"

            user_data.sock.sendto(message.encode(), (user_data.fpga_ip, user_data.port))

            string_to_print += (f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}"
				                f" Center:({center_x:.4f}, {center_y:.4f})\n"
                                f" Distance: {distance:.2f}")
            print(f"Sent message: {message}")

            detection_count += 1

    if user_data.use_frame:
        # Note: using imshow will not work here, as the callback function is not running in the main thread
        # Let's print the detection count to the frame
        cv2.putText(frame, f"Detections: {detection_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Example of how to use the new_variable and new_function from the user_data
        # Let's print the new_variable and the result of the new_function to the frame
        cv2.putText(frame, f"{user_data.new_function()} {user_data.new_variable}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Convert the frame to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    print(string_to_print)
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
