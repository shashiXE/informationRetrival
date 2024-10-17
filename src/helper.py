import streamlit as st
import PyPDF2
from transformers import BertTokenizer, BertModel, pipeline
import numpy as np
import torch

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Load the text generation model (e.g., GPT-2) for corrections
generator = pipeline('text-generation', model='gpt2', device=0 if torch.cuda.is_available() else -1)

def upload_pdf():
    """Function to upload a PDF file."""
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    return uploaded_file

def read_pdf(file):
    """Function to read a PDF file and return its content."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=1000):
    """Function to chunk text into smaller parts."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def embed_chunks(chunks):
    """Function to embed text chunks using BERT."""
    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():  # Disable gradient calculation
            outputs = model(**inputs)
        chunk_embedding = outputs.last_hidden_state.mean(dim=1).numpy()
        embeddings.append(chunk_embedding)
    return np.vstack(embeddings)

def query_model(query, embeddings, chunks):
    """Function to query the model and find the most relevant chunk."""
    inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        query_embedding = model(**inputs).last_hidden_state.mean(dim=1).numpy()
    
    similarities = np.dot(embeddings, query_embedding.T).flatten()
    most_similar_idx = np.argmax(similarities)
    return chunks[most_similar_idx], similarities[most_similar_idx]

def correct_response(query, initial_response, temperature):
    """Function to generate a corrected response using the LLM."""
    prompt = f"Correct the following response to the question '{query}': {initial_response}"
    corrected_response = generator(prompt, max_new_tokens=50, temperature=temperature, num_return_sequences=1)
    return corrected_response[0]['generated_text']