from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from create_db import generate_data_store, Chroma_path

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize components
llm_hf = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2", max_length=128, temperature=0.5, token=HUGGINGFACEHUB_API_TOKEN
)

def create_prompt(cnt, diffi, topic):
    return f"So you want to create an MCQ quiz with {cnt} questions with {diffi} difficulty level based on {topic.capitalize()}."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Collect other form data
        num_questions = request.form.get('num_questions')
        difficulty = request.form.get('difficulty')
        description = request.form.get('description')

        # Process the PDF and create embeddings
        generate_data_store()

        # Generate quiz
        prompt = create_prompt(num_questions, difficulty, description)
        quiz = generate_quiz(prompt)

        print(quiz)
        return render_template('quiz.html', quiz=quiz)


def generate_quiz(prompt):
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    docsearch = Chroma(
        persist_directory=Chroma_path,
        embedding_function=OpenAIEmbeddings()
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_hf,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )
    response = chain({"question": prompt}, return_only_outputs=True)
    answer = response.get("answer", "")
    
    return answer



if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(Chroma_path):
        os.makedirs(Chroma_path)
    app.run(debug=True)
