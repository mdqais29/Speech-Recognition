from speech_model import load_model, transcribe

def main():
    print("Loading speech recognition model...")
    processor, model = load_model()

    audio_path = input("Enter path to audio file (.wav): ").strip()

    print("\nTranscribing...\n")
    text = transcribe(processor, model, audio_path)

    print("===== TRANSCRIPTION =====\n")
    print(text)

if __name__ == "__main__":
    main()