from datasets import load_dataset

# Automatically download the english test subset of Mozilla commonvoice
# This will save somewhere to disk - identify the path and use this path as our directory from which we provide examples to whisper.cpp
# Model card: https://huggingface.co/datasets/fsicoli/common_voice_17_0
cv_17 = load_dataset("fsicoli/common_voice_17_0", "en", split="test")

# Accuracy measurement: Word Error Rate
# Use the following package: https://github.com/jitsi/jiwer
