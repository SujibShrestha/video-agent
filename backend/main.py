from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="AI Video Backend", version="1.0.0")


class ProcessRequest(BaseModel):
    source: str
    language: str = "en"


class ChatRequest(BaseModel):
    question: str
    transcript: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/process")
def process_video(payload: ProcessRequest):
    from utils.audio_processor import process_input
    from core.transcriber import transcribe_all
    from core.summarize import summarize, generate_title
    from core.extractor import extract_action_items, extract_key_decisions, extract_questions
    from core.rag_engine import bulid_rag_chain

    try:
        print("starting ai video assistant")
        chunks = process_input(payload.source)
        transcript = transcribe_all(chunks, payload.language)
        print(f"raw transcription (first 300 char) {transcript[:300]}")

        title = generate_title(transcript)
        summary = summarize(transcript)
        action_items = extract_action_items(transcript)
        decisions = extract_key_decisions(transcript)
        questions = extract_questions(transcript)
        bulid_rag_chain(transcript)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "decisions": decisions,
        "questions": questions,
        "chat_ready": True,
    }


@app.post("/chat")
def chat_with_video(payload: ChatRequest):
    from core.rag_engine import ask_question, bulid_rag_chain

    try:
        rag_chain = bulid_rag_chain(payload.transcript)
        answer = ask_question(rag_chain, payload.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"answer": answer}


def run_pipeline(source: str, language: str = "en"):
    from utils.audio_processor import process_input
    from core.transcriber import transcribe_all
    from core.summarize import summarize, generate_title
    from core.extractor import extract_action_items, extract_key_decisions, extract_questions
    from core.rag_engine import bulid_rag_chain, ask_question

    print("starting ai video assistant")

    chunks = process_input(source)
    transcript = transcribe_all(chunks, language)
    print(f"raw transcription (first 300 char) {transcript[:300]}")

    title = generate_title(transcript)
    summary = summarize(transcript)
    action_items = extract_action_items(transcript)
    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)
    rag_chain = bulid_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "decisions": decisions,
        "questions": questions,
        "rag_chain": rag_chain,
    }


def run_cli():
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english): ").strip() or "english"
    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['decisions']}")
    print(f"\n❓ Open Questions:\n{result['questions']}")
    print("=" * 60)

    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    from core.rag_engine import ask_question

    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\n🤖 Assistant: {answer}\n")


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true", help="Run the interactive CLI workflow")
    args = parser.parse_args()

    if args.cli:
        run_cli()
    else:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)