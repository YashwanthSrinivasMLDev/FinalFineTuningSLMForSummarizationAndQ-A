import torch

consolidated_pdf_text_chunks = []
consolidated_training_data_for_summarization  = []
consolidated_training_data_for_qa  = []
device = "cuda" if torch.cuda.is_available() else "cpu"
import json

def create_training_data_summary_data_for_if ():
    print("creating summarized training data ")
    with open("training_data/article-summaries/article_summary_pairs_array", "r") as f:
        article_summary_pairs = json.load(f)
    return article_summary_pairs
