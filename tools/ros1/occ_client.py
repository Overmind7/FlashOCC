#!/usr/bin/env python
import json
import time
import runpy

import cv2
import message_filters
import numpy as np
import requests
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import String


def encode_jpeg(cv_image, quality=90):
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buffer = cv2.imencode('.jpg', cv_image, encode_params)
    if not ok:
        raise RuntimeError('Failed to encode image to JPEG')
    return buffer.tobytes()


def _extract_voxel_config(cfg: dict):
    point_cloud_range = cfg.get("point_cloud_range")
    voxel_size = cfg.get("voxel_size")
    if point_cloud_range is None or voxel_size is None:
        grid_config = cfg.get("grid_config", {})
        if point_cloud_range is None:
            x_cfg = grid_config.get("x")
            y_cfg = grid_config.get("y")
            z_cfg = grid_config.get("z")
            if x_cfg and y_cfg and z_cfg:
                point_cloud_range = [x_cfg[0], y_cfg[0], z_cfg[0], x_cfg[1], y_cfg[1], z_cfg[1]]
        if voxel_size is None and grid_config:
            voxel_size = [
                grid_config.get("x", [0.0, 0.0, 1.0])[2],
                grid_config.get("y", [0.0, 0.0, 1.0])[2],
                grid_config.get("z", [0.0, 0.0, 1.0])[2],
            ]
    return point_cloud_range, voxel_size


def _load_voxel_config(config_path: str):
    cfg = runpy.run_path(config_path)
    return _extract_voxel_config(cfg)


def _occ_to_points(occ_map: np.ndarray, point_cloud_range, voxel_size):
    mask = occ_map != 4
    if not np.any(mask):
        return np.empty((0, 4), dtype=np.float32)
    idxs = np.column_stack(np.where(mask))
    x = point_cloud_range[0] + idxs[:, 0] * voxel_size[0]
    y = point_cloud_range[1] + idxs[:, 1] * voxel_size[1]
    z = point_cloud_range[2] + idxs[:, 2] * voxel_size[2]
    labels = occ_map[mask].astype(np.float32)
    return np.column_stack((x, y, z, labels)).astype(np.float32)


class OccClient(object):
    def __init__(self):
        self.server_url = rospy.get_param('~server_url', 'http://localhost:5801/infer')
        self.topic_left = rospy.get_param('~camera_left', '/camera_image_left')
        self.topic_right = rospy.get_param('~camera_right', '/camera_image_right')
        self.topic_front = rospy.get_param('~camera_front', '/camera_image_front')
        self.publish_topic = rospy.get_param('~occ_topic', '/occ')
        self.queue_size = rospy.get_param('~queue_size', 10)
        self.slop = rospy.get_param('~slop', 0.1)
        self.jpeg_quality = rospy.get_param('~jpeg_quality', 90)
        self.timeout = rospy.get_param('~timeout', 30.0)
        self.max_rate_hz = rospy.get_param('~max_rate_hz', 0.0)
        self.min_interval = 1.0 / self.max_rate_hz if self.max_rate_hz > 0 else 0.0
        self.last_request_time = 0.0

        self.pointcloud_topic = rospy.get_param('~occ_cloud_topic', '')
        self.pointcloud_frame = rospy.get_param('~occ_cloud_frame', 'map')
        self.point_cloud_range = rospy.get_param('~point_cloud_range', None)
        self.voxel_size = rospy.get_param('~voxel_size', None)
        config_path = rospy.get_param('~config_path', '')

        if self.pointcloud_topic and (self.point_cloud_range is None or self.voxel_size is None):
            if config_path:
                self.point_cloud_range, self.voxel_size = _load_voxel_config(config_path)
            else:
                rospy.logwarn('occ_cloud_topic set but missing voxel config; provide ~config_path or ~point_cloud_range/~voxel_size')

        self.bridge = CvBridge()
        self.publisher = rospy.Publisher(self.publish_topic, String, queue_size=10)
        self.pointcloud_publisher = None
        if self.pointcloud_topic and self.point_cloud_range and self.voxel_size:
            self.pointcloud_publisher = rospy.Publisher(self.pointcloud_topic, PointCloud2, queue_size=1)

        self.sub_left = message_filters.Subscriber(self.topic_left, Image)
        self.sub_right = message_filters.Subscriber(self.topic_right, Image)
        self.sub_front = message_filters.Subscriber(self.topic_front, Image)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_left, self.sub_right, self.sub_front],
            queue_size=self.queue_size,
            slop=self.slop,
        )
        self.sync.registerCallback(self.synced_callback)

        rospy.loginfo('OccClient initialized: %s, %s, %s -> %s',
                      self.topic_left, self.topic_right, self.topic_front, self.publish_topic)

    def synced_callback(self, left_msg, right_msg, front_msg):
        try:
            now = time.time()
            if self.min_interval > 0 and (now - self.last_request_time) < self.min_interval:
                rospy.logdebug_throttle(5.0, 'OccClient rate limited (max_rate_hz=%.2f)', self.max_rate_hz)
                return
            left_cv = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
            right_cv = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')
            front_cv = self.bridge.imgmsg_to_cv2(front_msg, desired_encoding='bgr8')

            left_rgb = cv2.cvtColor(left_cv, cv2.COLOR_BGR2RGB)
            right_rgb = cv2.cvtColor(right_cv, cv2.COLOR_BGR2RGB)
            front_rgb = cv2.cvtColor(front_cv, cv2.COLOR_BGR2RGB)

            files = {
                'image_left': ('left.jpg', encode_jpeg(left_rgb, self.jpeg_quality), 'image/jpeg'),
                'image_right': ('right.jpg', encode_jpeg(right_rgb, self.jpeg_quality), 'image/jpeg'),
                'image_front': ('front.jpg', encode_jpeg(front_rgb, self.jpeg_quality), 'image/jpeg'),
            }
            payload = {
                'stamp_left': left_msg.header.stamp.to_sec(),
                'stamp_right': right_msg.header.stamp.to_sec(),
                'stamp_front': front_msg.header.stamp.to_sec(),
            }
            if self.pointcloud_publisher is not None:
                payload['return_occ'] = True
            json_data = json.dumps(payload)

            start = time.time()
            response = requests.post(
                self.server_url,
                files=files,
                data={'json': json_data},
                timeout=self.timeout,
            )
            elapsed = time.time() - start
            response.raise_for_status()

            self.last_request_time = now

            result = response.json()
            result['latency_sec'] = elapsed
            msg = String(data=json.dumps(result, ensure_ascii=False))
            self.publisher.publish(msg)
            if self.pointcloud_publisher is not None and result.get('occ') is not None:
                occ_map = np.array(result['occ'], dtype=np.uint8)
                points = _occ_to_points(occ_map, self.point_cloud_range, self.voxel_size)
                fields = [
                    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                    PointField(name='label', offset=12, datatype=PointField.FLOAT32, count=1),
                ]
                header = rospy.Header()
                header.stamp = rospy.Time.now()
                header.frame_id = self.pointcloud_frame
                cloud_msg = point_cloud2.create_cloud(header, fields, points)
                self.pointcloud_publisher.publish(cloud_msg)
        except Exception as exc:
            rospy.logwarn('OccClient request failed: %s', exc)


if __name__ == '__main__':
    rospy.init_node('occ_client')
    OccClient()
    rospy.spin()
