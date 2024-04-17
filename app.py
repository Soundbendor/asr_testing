import logging
import subprocess

from datasets import load_dataset
from jiwer import wer

# Automatically download the english test subset of Mozilla commonvoice
# This will save somewhere to disk - identify the path and use this path as our directory from which we provide examples to whisper.cpp
# Model card: https://huggingface.co/datasets/fsicoli/common_voice_17_0
cv_17 = load_dataset("fsicoli/common_voice_17_0", "en", split="test")

# Accuracy measurement: Word Error Rate
# Use the following package: https://github.com/jitsi/jiwer

# error = wer(predicted_label, true_label)

# Transcription code provided by Will for interfacing with whisper.cpp
"""
Will Richards, Oregon State University, 2024

Abstraction layer for automated speech recognition (ASR) of recorded audio
"""

# TODO: Configure makefile such that we build the Whisper binary based on the desired model architecture
# This might involve some syscalls
# or, we just build everything once and then retrieve the binary from the assosciated models directory


class AudioTranscriber:
    def __init__(self, model="base.en-q5_0"):
        self.modelPath = f"../whisper.cpp/models/ggml-{model}.bin"

    def transcribe(self, inputFile: str):
        full_command = f"../whisper.cpp/main -m {self.modelPath} -f {inputFile} -np -nt"
        process = subprocess.Popen(
            full_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        # Get the output and error (if any)
        output, error = process.communicate()

        if error:
            logging.error(f"Error proccessing audio: {error.decode('utf-8')}")

        # Process and return the output string
        decoded_str = output.decode("utf-8").strip()
        processed_str = decoded_str.replace("[BLANK_AUDIO]", "").strip()

        return processed_str


def main():
    pass


if __name__ == "__main__":
    pass
