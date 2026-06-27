from utils.audio_processor import process_input
from core.transcriber import transcribe_all, translate_transcript

source = "https://youtu.be/KUwFPbgvzMA?si=dTMEflbsFlg7emE3"

chunks = process_input(source)

transcript = transcribe_all(chunks)

translated = translate_transcript(transcript,target_language="en")
print(translated)