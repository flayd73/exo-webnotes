import sys
import whisper
import argparse

def transcribe_audio(audio_path, model_name="base", output_file=None):
    # Load the model
    print(f"Loading {model_name} model...")
    model = whisper.load_model(model_name)
    
    # Actually do the transcription
    print(f"Transcribing {audio_path}...")
    result = model.transcribe(audio_path)
    
    # Get the text
    text = result["text"]
    
    # Write to file or print to console
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Transcription saved to {output_file}")
    else:
        print("\nTranscription:")
        print(text)
    
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio using Whisper")
    parser.add_argument("audio_path", help="Path to the audio file")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"], 
                        help="Model size (tiny, base, small, medium, large)")
    parser.add_argument("--output", help="Output file path (optional)")
    
    args = parser.parse_args()
    
    transcribe_audio(args.audio_path, args.model, args.output) 