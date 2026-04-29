#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

import gi
import numpy as np
import cv2
from ultralytics import YOLO

gi.require_version("Gst", "1.0")
from gi.repository import Gst


class VideoInterfaceNode(Node):
    def __init__(self):
        super().__init__("video_interface")

        self.position_pub = self.create_publisher(Point, "/object_position", 10)

        self.model = YOLO("yolov8n.pt")  # COCO class 0 = person

        self.declare_parameter(
            "gst_pipeline",
            (
                'udpsrc port=5000 caps="application/x-rtp,media=video,'
                'encoding-name=H264,payload=96" ! '
                "rtph264depay ! avdec_h264 ! videoconvert ! "
                "video/x-raw,format=RGB ! appsink name=sink"
            ),
        )

        pipeline_str = self.get_parameter("gst_pipeline").value

        Gst.init(None)
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.sink = self.pipeline.get_by_name("sink")

        self.sink.set_property("drop", True)
        self.sink.set_property("max-buffers", 1)

        self.pipeline.set_state(Gst.State.PLAYING)

        self.timer = self.create_timer(1.0 / 30.0, self.on_timer)

        self.get_logger().info("VideoInterfaceNode initialized with YOLOv8 person tracking")

    def on_timer(self):
        sample = self.sink.emit("pull-sample")
        if not sample:
            return

        buf = sample.get_buffer()
        caps = sample.get_caps()

        width = caps.get_structure(0).get_value("width")
        height = caps.get_structure(0).get_value("height")

        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return

        frame = np.frombuffer(mapinfo.data, np.uint8).reshape(height, width, 3).copy()
        buf.unmap(mapinfo)

        results = self.model(frame, verbose=False)[0]

        best_box = None
        best_conf = 0.0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # 0 -> class for person in COCO dataset
            if cls_id == 0 and conf > best_conf and conf > 0.5:
                best_conf = conf
                best_box = box

        point = Point()
        debug_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if best_box is not None:
            x1, y1, x2, y2 = best_box.xyxy[0].tolist()

            center_x = (x1 + x2) / 2.0
            area = np.abs(x2 - x1) * np.abs(y2 - y1)

            point.x = center_x
            point.y = 0.0 # don't care
            point.z = area

            cv2.rectangle(
                debug_frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                debug_frame,
                f"person conf={best_conf:.2f} area={area:.0f}",
                (int(x1), max(int(y1) - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        else:
            point.x = -1.0
            point.y = 0.0
            point.z = 0.0

            cv2.putText(
                debug_frame,
                "No person detected",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        self.position_pub.publish(point)

        cv2.imshow("RELBot YOLOv8 Person Tracking", debug_frame)
        cv2.waitKey(1)

    def destroy_node(self):
        self.pipeline.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VideoInterfaceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()