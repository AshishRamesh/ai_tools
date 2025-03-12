import rclpy,time,os , cv2, base64,pygame
from rclpy.node             import Node
from geometry_msgs.msg      import Point
from geometry_msgs.msg      import Twist
from openai                 import OpenAI
from sensor_msgs.msg        import Image
from cv_bridge              import CvBridge, CvBridgeError
from dotenv                 import load_dotenv
from PIL                    import Image as PILImage
from gtts                   import gTTS
from io                     import BytesIO

load_dotenv()

class Prompt(Node):
    def __init__(self):
        super().__init__('prompt_engine')
        pygame.init()
        pygame.mixer.init()
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_sub = self.create_subscription(Image,"/camera/image_raw/uncompressed",self.image_callback,rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value)
        self.bridge = CvBridge()
        self.cv_image = None
        self.client = OpenAI(base_url="https://models.inference.ai.azure.com",api_key=os.getenv('API_KEY'))

        self.timer = self.create_timer(1.0, self.process_voice_command)

    def get_direction(self, obstacle_direction):
        """Gets movement direction from the AI model."""
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "Assume you are a robot and have the capability of giving the commands: front, back, left, right, capture . "
                                   "Capture is used to capture an image and describe it "
                                   "Based on these, tell me what action you would take for the upcoming scenarios. "
                                   "I want only the action as the response, nothing else."
                    },
                    {
                        "role": "user",
                        "content": obstacle_direction,
                    }
                ],
                model="gpt-4o-mini",
                max_tokens=5,
                n=1
            )
            return response.choices[0].message.content.strip().lower()
        except Exception as e:
            self.get_logger().error(f"Error getting direction from AI: {e}")
            return "stop"

    def image_callback(self, data):
        """Receives and converts camera images."""
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")  
            cv2.imshow("Image", self.cv_image)
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error: {e}")

    def move_robot(self, direction):
        """Moves the robot based on AI's response."""
        msg = Twist()
        
        if direction == "front":
            msg.linear.x = 30.5  
            self.get_logger().info("Moving forward")
        
        elif direction == "back":
            msg.linear.x = -3.5  
            self.get_logger().info("Moving backward")

        elif direction == "left":
            msg.angular.z = 3.0  
            self.get_logger().info("Turning left")

        elif direction == "right":
            msg.angular.z = -3.0  
            self.get_logger().info("Turning right")

        elif direction == "capture":
            image_file = self.capture_image()
            if image_file:
                self.send_image_for_description(image_file)

        else:
            self.get_logger().warn(f"Invalid direction received{direction}")
            return

        self.publisher_.publish(msg)
        time.sleep(5)  
        self.stop_robot()

    def stop_robot(self):
        """Stops the robot."""
        stop_msg = Twist()
        self.publisher_.publish(stop_msg)
        self.get_logger().info("Stopping robot")

    def process_voice_command(self):
        """Main loop to process voice commands continuously."""
        # obstacle_direction = self.recognize_speech()
        obstacle_direction = input("Enter the direction: ")
        if obstacle_direction:
            move_command = self.get_direction(obstacle_direction)
            self.move_robot(move_command)

    def capture_image(self, filename='captured_image.jpg'):
        """Captures an image from the ROS2 camera topic and resizes it before saving."""
        if self.cv_image is None:
            self.get_logger().error("No image received yet. Waiting for image...")
            rclpy.spin_once(self, timeout_sec=2.0)
            if self.cv_image is None:
                self.get_logger().error("Still no image received. Cannot capture.")
                return None

        try:
            resized_image = cv2.resize(self.cv_image, (640, 360))
            cv2.imwrite(filename, resized_image)
            self.get_logger().info(f"Image saved as {filename} (Resized to 640x360)")
            return filename
        except Exception as e:
            self.get_logger().error(f"Failed to save image: {e}")
            return None


    def encode_image(self, image_path):
        """Encodes an image to Base64 format for OpenAI API."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            self.get_logger().error(f"Error encoding image: {e}")
            return None

    def resize_and_compress_image(self,image_path, output_path='compressed_image.jpg', size=(320, 180), quality=80):
        """Resizes and compresses the image to reduce file size."""
        with PILImage.open(image_path) as img:
            img = img.resize(size, PILImage.LANCZOS) 
            img.save(output_path, "JPEG", quality=quality)  
        return output_path

    def send_image_for_description(self, image_file):
        """Sends the captured image to OpenAI's API for description."""
        if image_file:
            compressed_image = self.resize_and_compress_image(image_file)
        
        base64_image = self.encode_image(compressed_image)
        if not base64_image:
            return

        try:
            response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a Autonomous Mobile Robot and can describe what you see keep it short. "
                 "Also give response starting with I see and you dont have to mention if the image is blur or dim "},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=500,
            n=1,
            )
            description = response.choices[0].message.content.strip().lower()
            self.get_logger().info(f"Image Description: {description}")
            self.speak(description)
        except Exception as e:
            self.get_logger().error(f"Error during OpenAI API request: {e}")

    def speak(self,text, language='en'):
        mp3_fo = BytesIO()
        tts = gTTS(text, lang=language)
        tts.write_to_fp(mp3_fo)
        mp3_fo.seek(0)
        sound = pygame.mixer.Sound(mp3_fo)
        sound.play()
        self.wait_for_audio()

    def wait_for_audio(self):
        while pygame.mixer.get_busy():
            time.sleep(1)
    
    def move_cmd(self,liner=0.0,angular=0.0):
        msg = Twist()
        msg.linear.x = liner
        msg.angular.z = angular
        self.publisher_.publish(msg)
        time.sleep(5)
        self.stop_robot()

def main(args=None):
    rclpy.init(args=args)
    prompt_engine = Prompt()
    rclpy.spin(prompt_engine)  
    prompt_engine.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
