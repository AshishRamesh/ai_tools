# speech_to_text 
import whisper
import sounddevice as sd
import numpy as np
import wave
import keyboard  # For detecting spacebar press
import json
import os
from openai import OpenAI

# Load the Whisper model
model = whisper.load_model("base")

# Recording settings
SAMPLE_RATE = 44100  # Sampling rate (must be an integer)
FILENAME = "recorded_audio.wav"  # Output file

def record_audio():
    """Records audio while the spacebar is held down."""
    print("Press and hold SPACE to record...")

    frames = []
    
    def callback(indata, frames_, time, status):
        """Callback function to collect audio while recording."""
        if status:
            print(status)
        frames.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype=np.int16, callback=callback):
        while keyboard.is_pressed("space"):  # Keep recording while spacebar is pressed
            pass  # Do nothing, just wait

    print("Recording stopped. Processing...")

    # Save recorded audio to a WAV file
    if frames:
        with wave.open(FILENAME, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes per sample (16-bit audio)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(np.concatenate(frames).astype(np.int16).tobytes())

def transcribe_audio():
    """Transcribes the recorded audio using Whisper."""
    result = model.transcribe(FILENAME)
    transcribed_text = result["text"].strip()
    print("\nTranscribed Text:", transcribed_text)
    return transcribed_text

# Set up OpenAI client
client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ["MY_KEY"],
)

# Define available function calls
first_tools = [
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

# Function to interact with OpenAI GPT-4
def get_gpt_response(prompt):
    """Get structured response from OpenAI's GPT model."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are controlling a robot. Convert the given command into a structured JSON list of actions. Each action must have:"
                           " 'action' (mov_cmd or capture), 'linear_x' (distance), and 'angular_z' (angle in radians)."
                           " Return ONLY JSON, with no extra text."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        tools=first_tools,
    )

    gpt_text_response = response.choices[0].message.content.strip()

    print("GPT Raw Response:", gpt_text_response)  # Debugging: Print raw response

    # Ensure valid JSON extraction
    try:
        if gpt_text_response.startswith("```json"):
            gpt_text_response = gpt_text_response.strip("```json").strip("```")
        actions = json.loads(gpt_text_response)
    except json.JSONDecodeError as e:
        print("JSON Decode Error:", e)
        return None  # Return None to indicate failure

    return actions if isinstance(actions, list) else None

# Functions for executing commands
def move(distance, angle=0):
    """Handles movement."""
    return {"tool_call": "mov_cmd", "arguments": {"linear_x": distance, "angular_z": angle}}

def capture():
    """Handles capturing an image."""
    return {"tool_call": "capture", "arguments": {}}

def call_function_based_on_command(command):
    """Parse the transcribed command and call appropriate functions."""
    actions = get_gpt_response(command)

    print("Parsed Actions:", actions)  # Debugging: Print parsed actions

    if actions is None:
        return "Invalid response format from GPT."

    results = []
    for action in actions:
        action_type = action.get("action")
        if action_type == "mov_cmd":
            distance = action.get("linear_x", 0)
            angle = action.get("angular_z", 0)
            results.append(move(distance, angle))
        elif action_type == "capture":
            results.append(capture())

    return results if results else "No valid function found."

if __name__ == "__main__":
    print("\nPress SPACE to record, release to transcribe.")

    while True:
        keyboard.wait("space")  # Wait for spacebar press
        record_audio()
        command = transcribe_audio()  # Get transcribed text
        result = call_function_based_on_command(command)

        print("\nGenerated Robot Commands:\n")
        print(result)
