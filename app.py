#import neccessary libraries
import gradio as gr
import PyPDF2
from docx import Document
import pandas as pd
import os
import cohere
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Initialize Pinecone
from pinecone import Pinecone, ServerlessSpec
from pinecone import ServerlessSpec
from dotenv import load_dotenv
# Load environment variables from the .env file (if present)
load_dotenv()

cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'

spec = ServerlessSpec(cloud=cloud, region=region)



os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ['COHERE_API_KEY'] = os.getenv('COHERE_API_KEY')

pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

# Initialize Cohere client
co = cohere.Client(os.environ['COHERE_API_KEY'])

# Load SentenceTransformer model for encoding
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize question-answering pipeline
qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')


def load_and_preprocess_data(file_path,index_name, chunk_size=1000):
    if file_path is None:
        return []
    file_extension = os.path.splitext(file_path)[1].lower()
    data_chunks=""
    if file_extension == '.pdf':
        data_chunks = process_pdf(file_path, chunk_size)
    elif file_extension == ".docx":
        data_chunks = read_word_file(file_path)
    elif file_extension == '.csv':
        data_chunks = process_csv(file_path, chunk_size)
    elif file_extension in ['.txt', '.md']:
        data_chunks = process_text(file_path, chunk_size)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    encode_and_store(data_chunks,index_name)
    return data_chunks
# Function to read the content of a Word file
def read_word_file(file_path):
    doc = Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return "\n".join(full_text)

def process_pdf(file_path, chunk_size):
    chunks = []
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + " "

    # Split into chunks
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def process_csv(file_path, chunk_size):
    chunks = []
    df = pd.read_csv(file_path)
    for _, row in df.iterrows():
        row_text = " ".join(str(item) for item in row)
        chunks.extend([row_text[i:i+chunk_size] for i in range(0, len(row_text), chunk_size)])
    return chunks

def process_text(file_path, chunk_size):
    with open(file_path, 'r') as file:
        text = file.read()

    # Split into chunks
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def encode_and_store(chunks,indexName):
    # Check if there are exactly 3 indexes

    if len(pc.list_indexes()) == 3:
        for index in pc.list_indexes()[:2]:
            print(index)
            pc.delete_index(index.name)  # Delete the index using its name
            print(f"Deleting index '{index.name}'")

    # Create a Pinecone index (if it doesn't exist)
    index_name = indexName.lower().strip()
    # print(pc.list_indexes())
    for index in pc.list_indexes():
      if index_name == index.name.strip():
        index = pc.Index(index_name)
        print("Index already exists")
        break
    else:
      print("Creating index")
      pc.create_index(
        name=index_name,
        dimension=encoder.get_sentence_embedding_dimension(),
        metric="cosine",
        spec = spec
        )

    # Connect to the index
    index = pc.Index(index_name)

    # Encode and upsert data
    for i, chunk in enumerate(chunks):
        embedding = encoder.encode(chunk).tolist()
        print(f"Upserting chunk {i+1} with embedding of length {len(embedding)}")
        index.upsert([(str(i), embedding, {'text': chunk})])

# Encode and store the data
#encode_and_store(data_chunks)


def retrieve_relevant_chunks(query,index_name, top_k=10):
    index = pc.Index(index_name)
    query_embedding = encoder.encode(query).tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    #print(results)
    size = len(results['matches'])
    sum_of_score  = 0.0
    for result in results['matches']:
        sum_of_score+=result['score']
    avg_score = sum_of_score/size
    # print(avg_score)
    return [result['metadata']['text'] if result['score'] >=avg_score else "" for result in results['matches']]

# Function to format relevant chunks
def format_relevant_chunks(relevant_chunks):
    formatted_text = "\n\n".join(relevant_chunks)
    return formatted_text


def generate_answer(query, context):
    # Use Cohere for answer generation
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = co.generate(
        model='command-xlarge-nightly',
        prompt=prompt,
        max_tokens=350,
        temperature=0.7,
        #stop_sequences=["\n"]
    )
    return response.generations[0].text.strip()

def answer_question(query,index_name):
    relevant_chunks = retrieve_relevant_chunks(query,index_name)
    context = " ".join(relevant_chunks)
    answer = generate_answer(query, context)
    return answer

def chat(chat_history, query,index_name):
  # Step 1: Retrieve relevant chunks
  relevant_chunks = retrieve_relevant_chunks(query,index_name)
  relevant_chunks_text = f"Relevant indices: {format_relevant_chunks(relevant_chunks)}"
  bot_response = answer_question(query,index_name)
  yield relevant_chunks_text, chat_history + [(query, bot_response)]


def build_bot(pdf_file, index_name : str):
  try:
    file_data = load_and_preprocess_data(pdf_file, str(index_name).lower().strip())
    return "Successfully built bot...."
  except Exception as e:
    return f"Error: {str(e)}"



# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("### QA Bot - Upload a PDF and ask questions")

    # Upload PDF section
    pdf_file = gr.File(label="Upload PDF")
    gr.Markdown("Names can only contain lowercase letters, numbers, and hyphens (-).")
    index_name = gr.Textbox(label="Give a name to store you file")
    upload_status = gr.Textbox(label="Status...", lines=1)

    # Button to upload and process the PDF
    pdf_upload_btn = gr.Button("Build Bot")
    pdf_upload_btn.click(build_bot, inputs=[pdf_file, index_name], outputs=upload_status)

    relevant_chunks_output = gr.Textbox(label="Relevant Indices", lines=5)

    with gr.Tab("Knowledge Bot"):
        chatbot = gr.Chatbot()
        
        # Scaling the layout (80% Textbox, 20% Button)
        with gr.Row():
            with gr.Column(scale=8):  # 80% width for message
                message = gr.Textbox("What is this document about?", label="Question", lines=1)
            with gr.Column(scale=2):  # 20% width for button
                submit_btn = gr.Button("Submit")
        
        # When user presses Enter on the Textbox
        message.submit(chat, [chatbot, message, index_name], outputs=[relevant_chunks_output, chatbot])
        
        # When user clicks the Submit button
        submit_btn.click(chat, [chatbot, message, index_name], outputs=[relevant_chunks_output, chatbot])



#demo.queue().launch(debug = True)

demo.queue().launch(share=True)