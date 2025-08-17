import torch
from PyPDF2 import PdfReader
import json

consolidated_pdf_text_chunks = []
consolidated_training_data_for_summarization  = []
consolidated_training_data_for_qa  = []
device = "cuda" if torch.cuda.is_available() else "cpu"


def create_training_data_summary_data_for_if ():
    print("creating summarized training data ")
    with open("training_data/article-summaries/article_summary_pairs_array", "r") as f:
        article_summary_pairs = json.load(f)
    for pair in article_summary_pairs:
        pair['type']="summary"
    temp_qa_pair = []

    for pair in article_summary_pairs:
        # temp_qa_pair.append({'input': f"""You are a helpful AI assistant who summarizes
        # articles. Summarize the following article: {pair['article']}""" ,
        #             'target': f"Summary: {pair['summary']}", 'type': 'summary'})
        temp_qa_pair.append({'input': f"""You are a helpful AI assistant.
            Provide a concise summary of the following article:
            
            Article : 
            {pair['article']}""",
                             'target': f"Summary: {pair['summary']}", 'type': 'summary'})
    return temp_qa_pair

def create_training_data_qa_data_for_if ():
    print("creating qa training data ")
    with open("training_data/qa_data/training_data_qa_pairs", "r") as f:
        qa_pairs = json.load(f)
    temp_qa_pair = []
    # for pair in qa_pairs:
    #     temp_qa_pair.append({'input': f"""Task: Q&A.
    #     Input: User: {pair['question']}""" , 'target':f"Output: Agent: {pair['answer']}", 'type':'qa'})

    for pair in qa_pairs:
        temp_qa_pair.append({'input': f"""You are a helpful AI assistant. 
                Provide a concise and accurate answer to the following question. 
            
            Question:
                {pair['question']} """ ,
                             'target':f"Answer: {pair['answer']}", 'type':'qa'})


    return temp_qa_pair



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






