import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

MODEL_NAME = "facebook/wav2vec2-base-960h"

def load_model():
    print("Loading speech recognition model...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    return processor, model

def transcribe(processor, model, audio_path):
    speech, rate = librosa.load(audio_path, sr=16000)

    input_values = processor(
        speech,
        return_tensors="pt",
        padding="longest"
    ).input_values

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription.lower()