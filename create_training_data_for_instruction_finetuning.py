from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig, \
    AutoModelForSeq2SeqLM
import torch
import os
from datasets import load_dataset
from torch.utils.data import DataLoader
import pypdf

# model_id = "teknium/OpenHermes-2.5-Mistral-7B"
# model_id = "sshleifer/distilbart-cnn-12-6"
# model_id = "google/flan-t5-small"
consolidated_pdf_text_chunks = []
consolidated_training_data_for_summarization  = []
consolidated_training_data_for_qa  = []
device = "cuda" if torch.cuda.is_available() else "cpu"
import json

# def create_model_and_tokenizer_for_creating_training_data(model_id) :
#     # Set up quantization for memory efficiency
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16
#     )
#
#     # tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
#     # model = AutoModelForCausalLM.from_pretrained(
#     #     model_id,
#     #     quantization_config=bnb_config,
#     #     device_map="auto"
#     # )
#
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_id,
#                                                   # load_in_8bit = True ,
#                                                   device_map='auto',
#                                                   quantization_config=bnb_config)
#     return model,tokenizer




#creating chunks of text in a pdf to summarize
def chunk_text(text, chunk_size=200):
    words = text.split()
    chunks = []
    # Loop through the list of words in steps of chunk_size
    for i in range(0, len(words), chunk_size):
        # Join the words in each chunk back into a string
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def create_training_data_summary_data_for_if ():
    print("creating summarized training data ")
    with open("training_data/article-summaries/article_summary_pairs_array", "r") as f:
        article_summary_pairs = json.load(f)
    return article_summary_pairs
    # print("type article_summar_pairs", type(article_summary_pairs))
    # print("article_summary_pairs first object", article_summary_pairs[0])
    # print("article_summar_pairs", article_summary_pairs)
    # final_training_data = [f"""###Instruction:
    #                             Summarize the following article :
    #
    #                             {dict_article_summary['article']}
    #
    #                             ###Response
    #                             {dict_article_summary['summary']}
    #                                  """ for dict_article_summary in article_summary_pairs]
    # return final_training_data
    # base_dir = os.path.dirname(os.path.abspath(__file__))
    # Build the full path to the data directory
    # pdf_directory_path = os.path.join(base_dir, "training_data", "mock_health_insurance_policies")
    # pdf_directory_path = "./training_data/mock_health_insurance_policies"
    # pdf_directory_path = "./training_data/health_insurance_policies"
    # consolidated_chunks_of_pdf_text = []
    # for pdf_file in os.listdir(pdf_directory_path):
    #     if pdf_file.endswith(".pdf"):
    #         filepath = os.path.join(pdf_directory_path,pdf_file)
    #         pdf_text = ""
    #         reader = pypdf.PdfReader(filepath)
    #         for page in reader.pages:
    #             pdf_text += page.extract_text()
    #
    #         #create chunks of text for summarizing
    #         chunks = chunk_text(pdf_text)
    #         for chunk in chunks :
    #             # print("each chunk : ", chunk )
    #             consolidated_chunks_of_pdf_text.append(chunk)
    # summaries = summarize_lists_of_text(consolidated_chunks_of_pdf_text)
    # final_training_data = [(f"""###Instruction:
    #                         Summarize the following article :
    #
    #                         {consolidated_chunks_of_pdf_text[i]}
    #
    #                         ###Response
    #                         {summaries[i]}
    #                              """) for i in range(len(summaries))]

    # consolidated_training_data_for_summarization.append(summarized_value)
            # consolidated_pdf_text.append({"text": pdf_text})
    # print("final training data for IF summarization : ", final_training_data)
    # return final_training_data

# def create_prompt_template(text):
#     system_text = f"You are a helpful assistant who summarizes text"
#     user_text= f"Please summarize the following text {text}"
#
#     prompt = [
#         {"role":"system", "content": system_text},
#         {"role":"user", "content": user_text},
#     ]
#     return prompt
