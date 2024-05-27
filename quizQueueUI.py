from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
# import fitz  # PyMuPDF
from langchain.chains import (
    ConversationalRetrievalChain,
)
import PyPDF2
from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import os
import chainlit as cl



text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")


llm_hf = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=128, temperature=0.5, token= HUGGINGFACEHUB_API_TOKEN
)

def create_prompt(cnt, diffi, topic):
    return f"So you want to create an MCQ quiz with {cnt} no of questions with {diffi} difficulty level based on {topic.capitalize()}."

llm = llm_hf

@cl.on_chat_start
async def on_chat_start():

    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!",
            accept=["text/plain"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...", disable_feedback=True)
    await msg.send()

    with open(file.path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split the text into chunks
    texts = text_splitter.split_text(text)

    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a Chroma vector store
    embeddings = OpenAIEmbeddings()
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )

    message_history = ChatMessageHistory()
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )




    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. Now answer few questions for the Quiz"

    countt = await cl.AskActionMessage(
        content="Select the no of questions ",
        actions=[
            cl.Action(name="3", value="3", label="THREE"),
            cl.Action(name="5", value="5", label="FIVE"),
            cl.Action(name="7", value="7", label="SEVEN"),
        ],
    ).send()

    ginti = countt.get("value")

    Difficulty = await cl.AskActionMessage(
        content="Select Difficulty of Questions ",
        actions=[
            cl.Action(name="1", value="Easy", label="Easy"),
            cl.Action(name="2", value="Medium", label="Medium"),
            cl.Action(name="3", value="Hard", label="Hard"),
        ],
    ).send()

    diff = Difficulty.get("value")

    namee = await cl.AskUserMessage(content="Provide a name for the quiz?", timeout=100).send()
    if namee:
        await cl.Message(
            content=f"Your name is: {namee['output']}",
        ).send()

    desc = await cl.AskUserMessage(content="What should the quiz be based on?", timeout=100).send()
    if desc:
        await cl.Message(
            content=f"Your topic is: {desc['output']}",
        ).send()

    promptt = create_prompt(ginti,diff,desc['output'])


    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        llm = llm_hf,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
        # combine_docs_chain_kwargs={"prompt": promptt}
    )

    # print(promptt)
    
    msg = cl.Message(content=promptt, disable_feedback=True)
    await msg.send()

    msg = cl.Message(content="To confirm copy and paste the last message in the chat", disable_feedback=True)
    await msg.send()

    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()


    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    # await cl.Message(content=answer, elements=text_elements).send()
    await cl.Message(content=answer).send()