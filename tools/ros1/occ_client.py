#!/usr/bin/env python
import json
import time

import cv2
import message_filters
import requests
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String


def encode_jpeg(cv_image, quality=90):
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buffer = cv2.imencode('.jpg', cv_image, encode_params)
    if not ok:
        raise RuntimeError('Failed to encode image to JPEG')
    return buffer.tobytes()


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

        self.bridge = CvBridge()
        self.publisher = rospy.Publisher(self.publish_topic, String, queue_size=10)

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
            left_cv = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
            right_cv = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')
            front_cv = self.bridge.imgmsg_to_cv2(front_msg, desired_encoding='bgr8')

            files = {
                'image_left': ('left.jpg', encode_jpeg(left_cv, self.jpeg_quality), 'image/jpeg'),
                'image_right': ('right.jpg', encode_jpeg(right_cv, self.jpeg_quality), 'image/jpeg'),
                'image_front': ('front.jpg', encode_jpeg(front_cv, self.jpeg_quality), 'image/jpeg'),
            }
            payload = {
                'stamp_left': left_msg.header.stamp.to_sec(),
                'stamp_right': right_msg.header.stamp.to_sec(),
                'stamp_front': front_msg.header.stamp.to_sec(),
            }
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

            result = response.json()
            result['latency_sec'] = elapsed
            msg = String(data=json.dumps(result, ensure_ascii=False))
            self.publisher.publish(msg)
        except Exception as exc:
            rospy.logwarn('OccClient request failed: %s', exc)


if __name__ == '__main__':
    rospy.init_node('occ_client')
    OccClient()
    rospy.spin()
