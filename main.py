from transformers import pipeline
import streamlit as st
import fitz  # PyMuPDF

def parser(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:  # Adjust 'filetype' based on actual file type if needed
        for page in doc:
            text += page.get_text()
    return text

# Initialize the pipeline for summarization
pipe = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Streamlit UI
st.header("Reading's for nerds :nerd_face:")
st.subheader("JK, reading is cool")
st.write("This lil' app is a demonstration of Hugging Face's :hugging_face: transformers library")
st.write("This is my document summarizer! Insert documents and let your pc do the legwork of the reading by providing a short summarization, enjoy!")

file = st.file_uploader("Drag or drop your documents here!", type=['pdf'], accept_multiple_files=False)

if st.button("Submit") and file is not None:
    text_to_summarize = parser(file)
    if len(text_to_summarize) > 0:
        summary = pipe(text_to_summarize, max_length=130, min_length=30, do_sample=False)
        st.write(summary[0]['summary_text'])
    else:
        st.write("The document is empty or could not be read.")
