# PDF Query Application using Hugging Face Transformers

This project is a Streamlit-based web application that allows users to upload PDF files, process them into chunks, generate embeddings using a pre-trained BERT model, and ask questions about the PDF content. The application searches the PDF text for the most relevant chunks based on the user's query. Additionally, it includes a correction feature that uses a text generation model (GPT-2) to improve responses when necessary.

## Features

- **PDF Upload**: Users can upload a PDF file to the app, which extracts and processes the text.
- **Text Chunking**: The extracted text is split into smaller chunks to allow more effective querying.
- **BERT Embedding**: The application uses a BERT-based model (`bert-base-uncased`) to generate embeddings for the text chunks.
- **Query Functionality**: Users can ask questions related to the content of the uploaded PDF, and the app returns the most relevant chunk based on semantic similarity.
- **Response Correction**: If the response is not accurate, users can trigger a correction using GPT-2, with control over the temperature parameter to influence the variability of the generated text.
- **Chat History**: The app maintains a history of queries and responses during the session.

## Installation

1. Clone the repository:
   git clone
   cd pdf-query-app

2. pip install -r requirements.txt

3. streamlit run app.py

## Usage

### Uploading a PDF:
- Launch the app.
- Click the "Upload a PDF" button and select a PDF file from your local system.

### Chunking and Embedding:
- After uploading the PDF, the text is automatically extracted and divided into smaller chunks.
- The app will generate embeddings for each chunk using the BERT model.

### Asking a Question:
- Enter a query into the text input box.
- The app will display the most relevant chunk from the PDF based on the similarity between your query and the text embeddings.

### Response Correction:
- If the initial response is incorrect, select "No" when prompted.
- Adjust the temperature slider to control the level of creativity in the response.
- Click "Generate Correction" to get a refined response generated by GPT-2.

### Chat History:
- The application keeps track of all your queries and the corresponding responses during the session. You can review them under the "Chat History" section.

## Helper Functions

The code includes several helper functions, which are stored in the `src/helper.py` file:

- `upload_pdf()`: Handles file upload via Streamlit's file uploader widget.
- `read_pdf()`: Extracts text from the uploaded PDF using PyPDF2.
- `chunk_text(text, chunk_size)`: Divides the extracted text into smaller chunks.
- `embed_chunks(chunks)`: Uses the BERT model to generate embeddings for each text chunk.
- `query_model(query, embeddings, chunks)`: Finds the most relevant chunk for the user's query based on cosine similarity between the query embedding and the chunk embeddings.
- `correct_response(query, initial_response, temperature)`: Uses GPT-2 to correct or improve the initial response.

## Python Libraries

- **streamlit**: For building the web interface.
- **transformers**: For working with pre-trained Hugging Face models (BERT and GPT-2).
- **torch**: For running the BERT and GPT-2 models.
- **PyPDF2**: For reading the PDF content.
- **numpy**: For handling numerical operations such as calculating similarity scores.


# Prerequisites for Running the PDF Query Application

Before running the PDF Query Application using BERT and GPT-2 models, you'll need to set up the required environment and dependencies. Below are the key prerequisites:

---

## 1. Python Setup

Ensure that you have **Python 3.8 or above** installed on your machine. You can check your Python version by running the following command:

python --version


For macOS/Linux:

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt


Optional: Creating a .env File for API Keys
If you later decide to integrate API-based models like GPT-3 or GPT-4 from OpenAI or other providers, you may want to manage your API keys securely. You can create a .env file in the root directory and store your keys there. For example:

makefile
Copy code
OPENAI_API_KEY=your_openai_api_key
Use python-dotenv to load environment variables in the app:

bash
Copy code
pip install python-dotenv
Then in your code, you can load the API key:

python
Copy code
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

