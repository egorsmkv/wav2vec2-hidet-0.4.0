import torch
import soundfile as sf

from transformers import AutoModelForCTC, Wav2Vec2BertProcessor

from importlib.metadata import version

# Check versions of libraries
print("torch:", version("torch"))
print("transformers:", version("transformers"))
print("hidet:", version("hidet"))


# Config
model_name = "Yehor/w2v-bert-2.0-uk-v2"
device = "cuda:0"  # or cpu
sampling_rate = 16_000

# Load the model
asr_model = AutoModelForCTC.from_pretrained(model_name).to(device)
processor = Wav2Vec2BertProcessor.from_pretrained(model_name)

asr_model_opt = torch.compile(asr_model, backend="hidet")

# Files
paths = [
    "short_1_16k.wav",
]

# Extract audio
audio_inputs = []
for path in paths:
    audio_input, _ = sf.read(path)
    audio_inputs.append(audio_input)

# Transcribe the audio
inputs = processor(audio_inputs, sampling_rate=sampling_rate).input_features
features = torch.tensor(inputs).to(device)

print(features[:50])

# Inference
with torch.no_grad():
    logits = asr_model_opt(features).logits

predicted_ids = torch.argmax(logits, dim=-1)
predictions = processor.batch_decode(predicted_ids)

# Log results
print("Predictions:")
print(predictions)
