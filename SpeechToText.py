import whisper
from transformers import pipeline
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioPhraseAnalyzer:
    def __init__(self, whisper_model: str = "base", device: str = "cuda"):
        """Initialize the audio phrase analyzer with specified models."""
        try:
            self.whisper_model = whisper.load_model(whisper_model, device=device)
            self.nlp = pipeline(
                "zero-shot-classification", 
                model="facebook/bart-large-mnli", 
                device=0 if device == "cuda" else -1
            )
            logger.info("Models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
            
    def transcribe_and_find_phrases(self, 
                                  audio_file_path: str, 
                                  key_phrases: List[str], 
                                  confidence_threshold: float = 0.5
                                 ) -> Tuple[str, Dict[str, List[float]]]:
        """
        Transcribe audio and find timestamps of key phrases.
        
        Args:
            audio_file_path: Path to the audio file
            key_phrases: List of phrases to search for
            confidence_threshold: Minimum confidence score for phrase matching
            
        Returns:
            Tuple containing transcription text and timestamps dictionary
        """
        try:
            # Transcribe audio with progress indication
            logger.info(f"Transcribing audio file: {audio_file_path}")
            result = self.whisper_model.transcribe(audio_file_path, verbose=False)
            transcription = result['text']
            segments = result['segments']
            
            # Initialize timestamps dictionary
            timestamps = {phrase: [] for phrase in key_phrases}
            
            # Process segments with progress bar
            logger.info("Analyzing segments for key phrases")
            for segment in tqdm(segments, desc="Processing segments"):
                segment_text = segment['text'].lower()
                start_time = segment['start']
                
                # Batch process phrases for this segment
                results = self.nlp(segment_text, candidate_labels=key_phrases, multi_label=True)
                
                # Check each phrase result
                for phrase, score in zip(results['labels'], results['scores']):
                    if score > confidence_threshold:
                        timestamps[phrase].append(start_time)
                        
            return transcription, timestamps
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            raise
            
    def get_model_info(self) -> Dict[str, str]:
        """Return information about loaded models."""
        return {
            "whisper_model": "base",
            "nlp_model": "facebook/bart-large-mnli"
        }

def main():
    """Main execution function."""
    try:
        # Initialize analyzer
        analyzer = AudioPhraseAnalyzer()
        
        # Example audio file and phrases
        audio_file_path = "Resources/ReelGen Test Audio.m4a"
        key_phrases = [
            "what a goal", "and he's taking a free kick now", "it's a penalty",
            "he's been sent off", "a brilliant save", "he's scored",
            "it's a hat-trick", "the referee has blown the whistle",
            "it's a corner kick", "he's offside", "a fantastic header",
            "a stunning strike", "the goalkeeper is beaten", "it's an own goal",
            "a great tackle", "he's injured", "the match is over", "it's a draw",
            "they've won the match", "a beautiful pass", "he's dribbling past defenders",
            "a powerful shot", "the crowd is going wild", "a controversial decision",
            "the VAR is checking", "it's a yellow card", "it's a red card",
            "the captain is leading by example", "a last-minute goal",
            "the defense is solid", "a brilliant assist", "a counter-attack",
            "the manager is furious", "the fans are celebrating", "a crucial interception",
            "a long-range effort", "a quick throw-in", "a tactical substitution",
            "the team is pressing high"
        ]
        
        # Process audio
        transcription, timestamps = analyzer.transcribe_and_find_phrases(
            audio_file_path, 
            key_phrases,
            confidence_threshold=0.6  # Slightly higher threshold for better precision
        )
        
        # Print results
        print("Transcription:", transcription)
        print("\nKey Phrases Found:")
        for phrase, times in timestamps.items():
            if times:
                print(f"'{phrase}': {', '.join(f'{t:.2f}s' for t in times)}")
                
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()