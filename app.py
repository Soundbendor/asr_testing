import logging
import os
import subprocess
import time

import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset
from jiwer import wer

# Automatically download the english test subset of Mozilla commonvoice
# This will save somewhere to disk - identify the path and use this path as our directory from which we provide examples to whisper.cpp
# Model card: https://huggingface.co/datasets/fsicoli/common_voice_17_0
# TODO: Make sure this is coming in as 16k sample rate
cv_17 = load_dataset(
    "mozilla-foundation/common_voice_17_0", 
    "en", 
    split="test", 
    cache_dir="./dataset_cache", 
    streaming=True, 
    token=True
)
cv_17 = cv_17.cast_column("audio", Audio(sampling_rate=16000))
model_list = [
    "tiny.en",
    "base.en",
    "small.en",
    "medium.en"
]
"""
Will Richards, Oregon State University, 2024

Abstraction layer for automated speech recognition (ASR) of recorded audio
"""

class AudioTranscriber:
    def __init__(self, model="base.en-q5_0"):
        self.modelPath = f"whisper.cpp/models/ggml-{model}.bin"

    def transcribe(self, inputFile: str):
        full_command = f"whisper.cpp/main -m {self.modelPath} -f {inputFile} -np -nt"
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


# TODO: randomly sample the test subset... because there are too many damn samples



def compile_whisper_cpp() -> None:
# First, make sure we've compiled the quantization tool
    if not os.path.exists("whisper.cpp/main"):
        cmd = "cd whisper.cpp && make -j"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
        print(output)
    if not os.path.exists("whisper.cpp/quantize"):
        cmd = "cd whisper.cpp && make -j quantize"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
        print(output)
    # For every model, compile it, then quantize it
    for model in model_list:
        _compile_whisper_model(model)
        _quantize_whisper_model(model)

def _compile_whisper_model(model: str) -> None:
    if not os.path.exists(f"whisper.cpp/models/ggml-{model}.bin"):
        cmd = f"cd whisper.cpp && make -j {model}"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
        print(output)


def _quantize_whisper_model(model: str) -> None:
    if not os.path.exists(f"whisper.cpp/models/ggml-{model}-q5_0.bin"):
        cmd = f"cd whisper.cpp && quantize models/ggml-{model}.bin models/ggml-{model}-q5_0.bin q5_0"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
        print(output)


def main():
    # Ensure all model binaries are compiled... will skip if already available
    compile_whisper_cpp()
    models = model_list + [f"{x}-q5_0" for x in model_list]
    # For each model, run transcription experiment
    for model in models:
        result_df = test_transcription(model)


def test_transcription(model: str) -> pd.DataFrame:
    asr = AudioTranscriber(model=model)
    records = []
    for sample in cv_17.take(1000):
        print(sample)
        # Write sample to file
        basename = os.path.splitext(sample['path'])[0]
        sample_path = f"{basename}.wav"
        if not os.path.exists(sample_path):
            sf.write(sample_path, sample['audio']['array'], sample['audio']['sampling_rate'])
        # Dispatch file to ASR model for testing
        time_initial = time.time()
        pred = asr.transcribe(sample_path)
        time_final = time.time()
        # Compute WER between prediction and actual
        err = wer(sample['sentence'], pred)
        time_delta = time_final - time_initial
        records.append({"sample": sample['sentence'], "prediction": pred, "WER": err, "runtime": time_delta})
        
    # TODO: store this in a pd dataframe somwehre
    df = pd.DataFrame.from_records(records)
    df.name = model
    return df

if __name__ == "__main__":
    main()
