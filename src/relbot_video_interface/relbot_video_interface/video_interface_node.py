#!/usr/bin/env python3
import math

import cv2
import gi
import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from ultralytics import YOLO

gi.require_version("Gst", "1.0")
from gi.repository import Gst


class VideoInterfaceNode(Node):
    def __init__(self):
        super().__init__("video_interface")

        self.position_pub = self.create_publisher(Point, "/object_position", 10)

        self.model = YOLO("yolov8n.pt")
        self.declare_parameter("min_detection_confidence", 0.5)
        self.declare_parameter("person_class_id", 0)
        self.declare_parameter("base_bounding_area", 10000)

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
        self.min_detection_confidence = float(
            self.get_parameter("min_detection_confidence").value
        )
        self.person_class_id = int(self.get_parameter("person_class_id").value)
        self.base_bounding_area = int(self.get_parameter("base_bounding_area").value)
        self.bounding_scale = 10000 / self.base_bounding_area

        Gst.init(None)
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.sink = self.pipeline.get_by_name("sink")

        self.sink.set_property("drop", True)
        self.sink.set_property("max-buffers", 1)

        self.pipeline.set_state(Gst.State.PLAYING)

        self.timer = self.create_timer(1.0 / 30.0, self.on_timer)

        self.get_logger().info(
            "VideoInterfaceNode initialized with YOLOv8 + ByteTrack person tracking"
        )

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

        results = self.model.track(
            frame,
            persist = True,
            verbose = False,
            tracker = "bytetrack.yaml",
        )[0]

        tracked_box = None
        tracked_conf = 0.0
        tracked_id = None
        tracked_area = 0.0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if box.id is None:
                continue

            box_track_id = int(box.id[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            box_area = np.abs(x2 - x1) * np.abs(y2 - y1)

            if (
                    cls_id == self.person_class_id
                    and conf > self.min_detection_confidence
                    and box_area > tracked_area
            ):
                tracked_conf = conf
                tracked_box = box
                tracked_id = box_track_id
                tracked_area = box_area

        point = Point()
        debug_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image_center_x = width / 2.0
        cv2.line(
            debug_frame,
            (int(image_center_x), 0),
            (int(image_center_x), height),
            (255, 255, 0),
            2,
        )

        if tracked_box is not None:
            x1, y1, x2, y2 = tracked_box.xyxy[0].tolist()

            person_center_x = (x1 + x2) / 2.0
            area = np.abs(x2 - x1) * np.abs(y2 - y1)
            image_center_x = width / 2.0

            if math.fabs(person_center_x - image_center_x) > 20:
                point.X = person_center_x
                point.Y = 0.0
                point.z = 9999.0
            else:
                point.x = person_center_x
                point.y = 0.0
                point.z = self.bounding_scale * area

            cv2.rectangle(
                debug_frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                debug_frame,
                (
                    f"person id={tracked_id} "
                    f"conf={tracked_conf:.2f} x={person_center_x:.1f}"
                ),
                (int(x1), max(int(y1) - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                debug_frame,
                f"bbox area={area:.0f} published z={point.z:.1f}",
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            self.position_pub.publish(point)
            cv2.imshow("RELBot YOLOv8 Person Tracking", debug_frame)

        cv2.waitKey(1)

    def destroy_node(self):
        self.pipeline.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args = None):
    rclpy.init(args = args)
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
