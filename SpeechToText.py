import whisper
import torch
from transformers import pipeline

print(torch.cuda.is_available())

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to: {device}")

# Load the Whisper model from OpenAI
model = whisper.load_model("base", device=device)

# Load a pre-trained transformer model for NLP tasks
nlp = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if device == "cuda" else -1)

# Function to transcribe audio and find key phrases
def transcribe_and_find_phrases(audio_file_path, key_phrases):
    # Transcribe the audio file
    result = model.transcribe(audio_file_path)
    
    # Extract the transcription and segments
    transcription = result['text']
    segments = result['segments']
    
    # Initialize a dictionary to store timestamps of key phrases
    timestamps = {phrase: [] for phrase in key_phrases}
    
    # Iterate over the segments to find key phrases based on context
    for segment in segments:
        for phrase in key_phrases:
            # Use the NLP model to analyze the context of the segment
            result = nlp(segment['text'], candidate_labels=[phrase])
            if result['labels'][0] == phrase and result['scores'][0] > 0.5:
                timestamps[phrase].append(segment['start'])
    
    return transcription, timestamps

# Example usage
if __name__ == "__main__":
    audio_file_path = "Resources/ReelGen Test Audio.m4a"
    key_phrases = [
        "what a goal", 
        "and he's taking a free kick now", 
        "it's a penalty", 
        "he's been sent off", 
        "a brilliant save", 
        "he's scored", 
        "it's a hat-trick", 
        "the referee has blown the whistle", 
        "it's a corner kick", 
        "he's offside", 
        "a fantastic header", 
        "a stunning strike", 
        "the goalkeeper is beaten", 
        "it's an own goal", 
        "a great tackle", 
        "he's injured", 
        "the match is over", 
        "it's a draw", 
        "they've won the match", 
        "a beautiful pass", 
        "he's dribbling past defenders", 
        "a powerful shot", 
        "the crowd is going wild", 
        "a controversial decision", 
        "the VAR is checking", 
        "it's a yellow card", 
        "it's a red card", 
        "the captain is leading by example", 
        "a last-minute goal", 
        "the defense is solid", 
        "a brilliant assist", 
        "a counter-attack", 
        "the manager is furious", 
        "the fans are celebrating", 
        "a crucial interception", 
        "a long-range effort", 
        "a quick throw-in", 
        "a tactical substitution", 
        "the team is pressing high"
    ]
    
    transcription, timestamps = transcribe_and_find_phrases(audio_file_path, key_phrases)
    
    print("Transcription:", transcription)
    print("Timestamps:", timestamps)