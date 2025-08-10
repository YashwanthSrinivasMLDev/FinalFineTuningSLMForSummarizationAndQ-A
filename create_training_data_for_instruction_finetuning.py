from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig, \
    AutoModelForSeq2SeqLM
import torch
import os
from datasets import load_dataset
from torch.utils.data import DataLoader
import pypdf

# model_id = "teknium/OpenHermes-2.5-Mistral-7B"
# model_id = "sshleifer/distilbart-cnn-12-6"
model_id = "google/flan-t5-small"
consolidated_pdf_text_chunks = []
consolidated_training_data_for_summarization  = []
consolidated_training_data_for_qa  = []
device = "cuda" if torch.cuda.is_available() else "cpu"


def create_model_and_tokenizer_for_creating_training_data(model_id) :
    # Set up quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     quantization_config=bnb_config,
    #     device_map="auto"
    # )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id,
                                                  # load_in_8bit = True ,
                                                  device_map='auto',
                                                  quantization_config=bnb_config)
    return model,tokenizer




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
    print("creating training data")
    # base_dir = os.path.dirname(os.path.abspath(__file__))
    # Build the full path to the data directory
    # pdf_directory_path = os.path.join(base_dir, "training_data", "mock_health_insurance_policies")
    # pdf_directory_path = "./training_data/mock_health_insurance_policies"
    pdf_directory_path = "./training_data/health_insurance_policies"
    consolidated_chunks_of_pdf_text = []
    for pdf_file in os.listdir(pdf_directory_path):
        if pdf_file.endswith(".pdf"):
            filepath = os.path.join(pdf_directory_path,pdf_file)
            pdf_text = ""
            reader = pypdf.PdfReader(filepath)
            for page in reader.pages:
                pdf_text += page.extract_text()

            #create chunks of text for summarizing
            chunks = chunk_text(pdf_text)
            for chunk in chunks :
                # print("each chunk : ", chunk )
                consolidated_chunks_of_pdf_text.append(chunk)
    summaries = summarize_lists_of_text(consolidated_chunks_of_pdf_text)
    final_training_data = [(f"""###Instruction: 
                            Summarize the following article : 

                            {consolidated_chunks_of_pdf_text[i]}

                            ###Response 
                            {summaries[i]}
                                 """) for i in range(len(summaries))]

    # consolidated_training_data_for_summarization.append(summarized_value)
            # consolidated_pdf_text.append({"text": pdf_text})
    print("final training data for IF summarization : ", final_training_data)
    return final_training_data

# def create_prompt_template(text):
#     system_text = f"You are a helpful assistant who summarizes text"
#     user_text= f"Please summarize the following text {text}"
#
#     prompt = [
#         {"role":"system", "content": system_text},
#         {"role":"user", "content": user_text},
#     ]
#     return prompt

def summarize_lists_of_text(list_of_chat_messages : [str]):
    # consolidated_messages_after_applying_template = []
    # consolidated_formatted_messages = []
    # for message in list_of_chat_messages :
    #     formatted_message = create_prompt_template(message)
    #     formatted_prompt_for_tokenizing = tokenizer.apply_chat_template(formatted_message,tokenize = False, add_generation_prompt=True)
    #     consolidated_formatted_messages.append(formatted_prompt_for_tokenizing)
    model_for_generating_summaries, tokenizer_for_generating_summaries = create_model_and_tokenizer_for_creating_training_data(
        model_id)
    tokenized_prompt = tokenizer_for_generating_summaries(list_of_chat_messages, return_tensors='pt', padding=True, truncation= True  )
    tokenized_prompt = { key : value.to(device) for key, value in tokenized_prompt.items()}
    print("generating summaries")
    outputs = model_for_generating_summaries.generate(
        **tokenized_prompt,
        max_new_tokens= 100,
        num_beams=4,
        early_stopping=True
    )
    # Decode all summaries in the batch
    decoded_summaries = tokenizer_for_generating_summaries.batch_decode(outputs, skip_special_tokens=True)
    # for summary in decoded_summaries :
    #     print("summary : ", summary[0:50])
    print("deleting distilbart and tokenizer from memory ")
    del model_for_generating_summaries
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return decoded_summaries
    # Decode and post-process the output
    # decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    # # Extract only the summary part from the full conversation
    # # The output will include the prompt, so we need to remove it
    # summary = decoded_output.split("<|im_start|>assistant")[-1].strip()
    # # print('summarized output  ', summary)
    # return summary

