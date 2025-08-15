import torch
from transformers import pipeline
from PyPDF2 import PdfReader
import textwrap
import json

consolidated_pdf_text_chunks = []
consolidated_training_data_for_summarization  = []
consolidated_training_data_for_qa  = []
device = "cuda" if torch.cuda.is_available() else "cpu"


def create_training_data_summary_data_for_if ():
    print("creating summarized training data ")
    with open("training_data/article-summaries/article_summary_pairs_array", "r") as f:
        article_summary_pairs = json.load(f)
    return article_summary_pairs

def create_training_data_qa_data_for_if ():
    print("creating qa training data ")
    with open("training_data/qa_data/training_data_qa_pairs", "r") as f:
        qa_pairs = json.load(f)
    return qa_pairs

# def create_training_data_qa_data_for_if() :
#     # Step 3: Load Question Generation model
#     qg_pipeline = pipeline("text2text-generation", model="iarfmoose/t5-base-question-generator")
#
#     # Step 4: Process PDF → Chunks → Q&A
#     pdf_text = extract_text_from_pdf("example.pdf")
#     chunks = chunk_text(pdf_text, chunk_size=400)
#
#     qa_pairs = []
#     for chunk in chunks:
#         result = qg_pipeline(chunk, max_length=512, num_return_sequences=1)
#         qa_pairs.append(result[0]['generated_text'])
#
#     return qa_pairs

# Step 1: Extract text from PDF using PyPDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Step 2: Chunk the text (so it fits model max length)
def chunk_text(text, chunk_size=400):
    sentences = text.split('. ')
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks






