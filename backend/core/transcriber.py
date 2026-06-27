import whisper
import os
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
load_dotenv()

WHISPER_MODEL = os.getenv("WHISPER_MODEL","small")
_model = None

def load_model():
    global _model

    if _model is None:
        print(f"Loading model....")
        _model = whisper.load_model(WHISPER_MODEL)
        print("whisper model loaded successfully")

    return _model

def transcribe_chunk(chunk_path:str,translate:bool = False)->str:

    model = load_model()

    task = "translate" if translate else "transcribe"

    result = model.transcribe(chunk_path, task= task)

    return result['text']

def transcribe_all(chunks:list,translate:bool = False )->str:
    full_transcript = ""

    for i , chunk in enumerate(chunks):
        print(f"Transcribing chunk {i+1}")
        text = transcribe_chunk(chunk,translate= translate)

        full_transcript += text + " "

    print("Transcription completed")

    return full_transcript

def translate_transcript(transcript: str, target_language: str = "en") -> str:
    

    translator = GoogleTranslator(source="auto", target=target_language)

    # deep_translator has a 5000 char limit per call — split if needed
    if len(transcript) <= 5000:
        return translator.translate(transcript)

    # Split into chunks of 5000 chars at sentence boundaries
    words       = transcript.split(". ")
    current     = ""
    translated  = []

    for sentence in words:
        if len(current) + len(sentence) < 5000:
            current += sentence + ". "
        else:
            translated.append(translator.translate(current.strip()))
            current = sentence + ". "

    if current.strip():
        translated.append(translator.translate(current.strip()))

    return " ".join(translated)