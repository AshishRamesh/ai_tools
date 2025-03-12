import rclpy
from rclpy.node             import Node
from sensor_msgs.msg        import Image
from geometry_msgs.msg      import Point
from cv_bridge              import CvBridge, CvBridgeError

class ImgPrompt(Node):

    def __init__(self):
        super().__init__('detect_ball')

        self.get_logger().info('Prompt Engine Started.....!!!')
        self.image_sub = self.create_subscription(Image,"/image_in",self.callback,rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value)
        self.bridge = CvBridge()

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)


def main(args=None):

    rclpy.init(args=args)

    img_eng = ImgPrompt()

    img_eng.destroy_node()
    rclpy.shutdown()

