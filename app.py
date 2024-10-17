import streamlit as st
from src.helper import upload_pdf, read_pdf, chunk_text, embed_chunks, query_model, correct_response

def main():
    st.title("PDF Query Application using Hugging Face Transformers")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    if 'chunks' not in st.session_state:
        st.session_state.chunks = None

    pdf_file = upload_pdf()

    if pdf_file is not None:
        text = read_pdf(pdf_file)
        st.write("PDF Content Length:", len(text))

        st.session_state.chunks = chunk_text(text)
        st.write("Number of Chunks:", len(st.session_state.chunks))

        st.session_state.embeddings = embed_chunks(st.session_state.chunks)
        st.write("Embeddings generated for each chunk.")

    query = st.text_input("Ask a question about the PDF:")

    if query:
        response, similarity = query_model(query, st.session_state.embeddings, st.session_state.chunks)
        
        # Store the query and response in chat history
        st.session_state.chat_history.append({"query": query, "response": response})

        # Display the initial response
        st.write("**Initial Response:**", response)
        st.write("**Similarity Score:**", similarity)

        # Ask user if they want to correct the response
        correct = st.radio("Is the response correct?", ("Yes", "No"))

        if correct == "No":
            temperature = st.slider("Set Temperature for Correction:", 0.0, 1.0, 0.7, 0.1)
            if st.button("Generate Correction"):
                corrected_response = correct_response(query, response, temperature)
                st.write("**Corrected Response:**", corrected_response)
                # Store the corrected response in chat history
                st.session_state.chat_history.append({"query": f"Correction for: {query}", "response": corrected_response})

    # Display chat history
    if st.session_state.chat_history:
        st.write("### Chat History:")
        for chat in st.session_state.chat_history:
            st.write(f"**User:** {chat['query']}")
            st.write(f"**Response:** {chat['response']}")
            st.write("---")

if __name__ == "__main__":
    main()
