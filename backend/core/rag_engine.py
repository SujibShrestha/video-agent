import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from core.vector_store import bulid_vector_store,get_retriever,load_vector_store
from dotenv import load_dotenv
load_dotenv()

def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile",temperature=0.3)

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def bulid_rag_chain(transcript:str):
    vector_store = bulid_vector_store(transcript)
    retriever = get_retriever(vector_store,k=4)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert meeting assistant with deep analytical skills.
You are given a transcript or summary of a meeting and your job is to answer the user's question accurately and concisely based only on the meeting content provided.

Guidelines:
- Answer only from the context provided, do not make up information
- If the answer is not found in the context, say "This was not discussed in the meeting"
- Be concise and direct
- If action items or decisions are relevant to the question, highlight them
- Use bullet points for lists or multiple points

Context from meeting:
{context}
"""),
    ("human", "{question}")
])
    rag_chain = (
        {"context":retriever |RunnableLambda(format_docs),
         "question":RunnablePassthrough()}| prompt| llm | StrOutputParser()
    )

    return rag_chain

def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retriever()
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert meeting assistant with deep analytical skills.
You are given a transcript or summary of a meeting and your job is to answer the user's question accurately and concisely based only on the meeting content provided.

Guidelines:
- Answer only from the context provided, do not make up information
- If the answer is not found in the context, say "This was not discussed in the meeting"
- Be concise and direct
- If action items or decisions are relevant to the question, highlight them
- Use bullet points for lists or multiple points

Context from meeting:
{context}
"""),
    ("human", "{question}")
])
    rag_chain = (
        {"context": retriever |RunnableLambda(format_docs),
         "question":RunnablePassthrough()} | prompt | llm | StrOutputParser()
    )

    return rag_chain

def ask_question(rag_chain,question:str):
    print(f"Question : {question}")

    answer = rag_chain.invoke(question)
    print(f"answer: {answer}")