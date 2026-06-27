from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough,RunnableLambda


def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile",temperature=0.3)

def split_transcript(transcript:str)->list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 3000,
        chunk_overlap = 200
    )
    return splitter.split_text(transcript)

def summarize(transcript:str)->str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system","Summarize this portion of meeting transcript concisely."),
            ("human","{text}"),
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    chunks_summaries = [map_chain.invoke({"text":chunk}) for chunk in chunks]

    combined = "\n\n".join(chunks_summaries)

    combined_prompt = ChatPromptTemplate.from_messages([
        ('system','You are an expert meeting summarizer. Combine these partial summaries'
        'into one final professional meeting summary in bullet points'),
        ('human','{text}')
    ])

    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | combined_prompt | llm | StrOutputParser()  #send data to another section
    )

    return combined_chain.invoke(combined)
 


def generate_title(transcript: str) -> str:
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at creating concise, descriptive titles for meeting transcripts. Return only the title, nothing else."),
        ("human", "Generate a short title for this transcript:\n\n{transcript}")
    ])

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({"transcript": transcript[:3000]})  # trim to avoid token limit