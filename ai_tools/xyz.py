import os
import whisper
import sounddevice as sd
import numpy as np
import wave
import pygame
import queue
import json
from dotenv import load_dotenv
from openai import OpenAI
import sys  # Needed for clean exit

# Load .env file
load_dotenv()
API_KEY = os.getenv("MY_KEY")

if not API_KEY:
    raise ValueError("API Key not found! Make sure .env file is set correctly.")

print("API Key found, proceeding with execution...")

# Initialize OpenAI Client
client = OpenAI(api_key=API_KEY)

# Load Whisper model
model = whisper.load_model("base")

# Audio Settings
SAMPLE_RATE = 44100
FILENAME = "recorded_audio.wav"

# Queue for live transcription
audio_queue = queue.Queue()
recording = False
transcribed_text = "Press SPACE to record, release to transcribe."

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Voice Command Recorder")

# Colors
BG_COLOR = (14, 17, 23)  # Dark background
TEXT_COLOR = (250, 250, 250)  # White text

# Font
font = pygame.font.Font(None, 30)


def draw_text(text):
    """Displays wrapped text on the Pygame window."""
    screen.fill(BG_COLOR)

    # Word wrapping
    words = text.split(" ")
    lines = []
    line = ""
    for word in words:
        test_line = line + word + " "
        if font.size(test_line)[0] < WIDTH - 40:
            line = test_line
        else:
            lines.append(line)
            line = word + " "
    lines.append(line)

    y = HEIGHT // 2 - (len(lines) * 15)  # Adjust text centering
    for line in lines:
        text_surface = font.render(line, True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=(WIDTH // 2, y))
        screen.blit(text_surface, text_rect)
        y += 30

    pygame.display.flip()


def audio_callback(indata, frames, time, status):
    """Puts recorded audio into a queue for live transcription."""
    if status:
        print(status)
    audio_queue.put(indata.copy())


def record_audio():
    """Records audio while SPACE is held and updates live transcription."""
    global recording, transcribed_text
    frames = []

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype=np.int16, callback=audio_callback):
        recording = True
        draw_text("Listening...")

        while recording:
            while not audio_queue.empty():
                frames.append(audio_queue.get())

            # Save temporary audio for live transcription
            if frames:
                with wave.open(FILENAME, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(np.concatenate(frames).astype(np.int16).tobytes())

                # Live transcription update
                try:
                    result = model.transcribe(FILENAME)
                    transcribed_text = result["text"].strip()
                    draw_text(transcribed_text)
                except Exception as e:
                    transcribed_text = "Error transcribing..."
                    print("Error:", e)


def transcribe_audio():
    """Final transcription after recording stops."""
    global transcribed_text
    draw_text("Processing...")

    try:
        result = model.transcribe(FILENAME)
        transcribed_text = result["text"].strip()
    except Exception as e:
        transcribed_text = "Error in transcription."
        print("Error:", e)

    draw_text(f"Understood: {transcribed_text}")
    print("Final Transcription:", transcribed_text)

    # Call OpenAI for function execution
    execute_robot_commands(transcribed_text)


# Define available function calls for OpenAI
available_functions = [
    {
        "type": "function",
        "function": {
            "name": "mov_cmd",
            "description": "Move the robot a specified distance and/or rotate to a given angle in radians.",
            "parameters": {
                "type": "object",
                "properties": {
                    "linear_x": {
                        "type": "number",
                        "description": "Distance to move. Positive for forward, negative for backward."
                    },
                    "angular_z": {
                        "type": "number",
                        "description": "Angle to turn in radians. Positive for left, negative for right."
                    }
                },
                "required": ["linear_x", "angular_z"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture",
            "description": "Capture an image using the camera.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


def get_gpt_response(prompt):
    """Get structured response from OpenAI's GPT model."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Convert voice commands into robot actions. Respond ONLY with valid function calls."},
            {"role": "user", "content": prompt}
        ],
        tools=available_functions,
        tool_choice="auto",
    )

    print("GPT Raw Response:", response)

    # Extract tool calls
    tool_calls = response.choices[0].message.tool_calls if response.choices[0].message.tool_calls else []

    return tool_calls


def execute_robot_commands(command):
    """Parse the transcribed command and execute functions."""
    actions = get_gpt_response(command)

    if not actions:
        print("No valid function calls detected.")
        return

    for action in actions:
        function_name = action.function.name
        arguments = json.loads(action.function.arguments)

        if function_name == "mov_cmd":
            linear_x = arguments.get("linear_x", 0)
            angular_z = arguments.get("angular_z", 0)
            print(f"Executing Movement: Distance = {linear_x}, Angle = {angular_z}")

        elif function_name == "capture":
            print("Executing Capture Command: Taking a picture.")


def clean_exit():
    """Ensures everything stops before exiting."""
    print("\nExiting cleanly...")
    pygame.quit()
    sys.exit()


def main():
    """Main event loop for Pygame UI."""
    global recording

    draw_text(transcribed_text)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                clean_exit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not recording:
                    recording = True
                    record_audio()

            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE and recording:
                    recording = False
                    transcribe_audio()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        clean_exit()
