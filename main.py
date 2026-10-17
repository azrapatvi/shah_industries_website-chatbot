from fastapi import FastAPI, Form
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_history_aware_retriever
from langchain.messages import HumanMessage,AIMessage,SystemMessage

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import os
from dotenv import load_dotenv

load_dotenv()

os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')
os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')

app=FastAPI()

loader = WebBaseLoader([
    "https://ratnagiri-scrap.netlify.app/",
    "https://ratnagiri-scrap.netlify.app/about",
    "https://ratnagiri-scrap.netlify.app/services",
    "https://ratnagiri-scrap.netlify.app/faq",
    "https://ratnagiri-scrap.netlify.app/contact_us"
])
data=loader.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=2000,chunk_overlap=500
)
splitted_text=splitter.split_documents(data)

embeddings = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
vector_store=FAISS.from_documents(splitted_text,embedding=embeddings)
retriever=vector_store.as_retriever()

llm=ChatGroq(model='openai/gpt-oss-120b')

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a friendly and helpful chatbot for Shah Industries,
        a scrap business in Ratnagiri.

        Rules:
        1. If the user says hi, hello, or greets you,
           reply naturally and politely.

        2. Answer customer questions using the provided website context.

        3. If the requested information is not mentioned on the website,
           politely explain that the details are not available on the website
           and suggest contacting the owner directly for more information.

        4. Display the owner's contact details only if they are available
           in the provided context. Include the owner's phone number,
           email address, or other contact information when available.

        5. If the owner's contact details are not available in the context,
           do not invent them. Tell the customer to check the website's
           Contact Us section for the owner's details.

        6. Never make up prices, services, business details, or contact
           information.

        7. Keep your answers simple, polite, and customer-friendly.

        Context:
        {context}
        """
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


document_chain=create_stuff_documents_chain(llm,prompt)

contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system","""rewrite the latest question as a standalone question using the chat_history."""),
    MessagesPlaceholder(variable_name='chat_history'),
    ("human","{input}")

])
history_retriever=create_history_aware_retriever(llm,retriever,contextualize_prompt)

rag_chain=create_retrieval_chain(history_retriever,document_chain)



chat_history = []

@app.get("/")
def home():
    return FileResponse("templates/index.html")


class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(request: ChatRequest):
    response = rag_chain.invoke({
        "input": request.message,
        "chat_history": chat_history
    })

    answer = response["answer"]

    chat_history.append(HumanMessage(content=request.message))
    chat_history.append(AIMessage(content=answer))

    return {"answer": answer}

