import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

class ImageSub(Node):
    def __init__(self):
        super().__init__('image_sub')
        self.count = 0
        self.sub = self.create_subscription(Image, '/camera/image_raw', self.cb, 10)

    def cb(self, msg: Image):
        self.count += 1
        if self.count % 30 == 0:  # log ~once per second at 30 Hz
            self.get_logger().info(
                f"Got image {self.count} | {msg.width}x{msg.height} | frame_id={msg.header.frame_id}"
            )

def main():
    rclpy.init()
    node = ImageSub()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
