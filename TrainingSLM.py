import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, BitsAndBytesConfig, \
    AutoModelForSeq2SeqLM
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import pypdf
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from accelerate import Accelerator
#using automatic mixed precision to reduce  memory usage
from torch.amp import autocast,GradScaler

model_state_dict_weights_path = "artifacts/model/model_weights.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
# BATCH_SIZE_CONTINUED_TRAINING = 4
BATCH_SIZE_CONTINUED_TRAINING = 1
# BATCH_SIZE_FINE_TUNING = 4
BATCH_SIZE_FINE_TUNING = 1

# base_model_for_this_project = "sshleifer/distilbart-cnn-12-6"
base_model_for_this_project = "TinyLlama/TinyLlama_v1.1"
tokenizer = AutoTokenizer.from_pretrained(base_model_for_this_project)

print("using device : ", device)

def tokenize_function(examples):
    # This function tokenizes the text data for the model
    return tokenizer(examples["text"], truncation=True, max_length=512)

def tokenize_instruction_finetuning(examples):
    # This function formats the data as an instruction for summarization fine-tuning
    prompt_template = "### Instruction:\nSummarize the following article:\n\n{article}\n\n### Summary:\n{summary}"
    texts = [prompt_template.format(article=art, summary=summ) for art, summ in zip(examples["article"], examples["summary"])]
    return tokenizer(texts, truncation=True, max_length=512)

def create_model():

    # using qlora peft for reduced memory usage
    # setting device

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )



    # Loading TinyLama from HuggingFace
    model = AutoModelForCausalLM.from_pretrained(base_model_for_this_project,
                                                 quantization_config=bnb_config,
                                                 device_map={"": 0})

    if tokenizer.pad_token is None:
        # Add a new special token for padding
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # Resize the model's token embeddings to include the new pad token
        model.resize_token_embeddings(len(tokenizer))

    return model

def load_insurace_datasets():
    # extracted policy information from pdfs found in this link : https://www.cms.gov/marketplace/resources/fact-sheets-faqs
    # pdf_directory_path = "./training_data/health_insurance_policies"
    # pdf_directory_path = "./training_data/mock_health_insurance_policies"
    pdf_directory_path = "./training_data/health_insurance_policies"
    consolidated_pdf_text = []
    for pdf_file in os.listdir(pdf_directory_path):
        if pdf_file.endswith(".pdf"):
            filepath = os.path.join(pdf_directory_path,pdf_file)
            pdf_text = ""
            reader = pypdf.PdfReader(filepath)
            for page in reader.pages:
                pdf_text += page.extract_text()
            consolidated_pdf_text.append({"text": pdf_text})

    data_dict = {'text': [doc['text'] for doc in consolidated_pdf_text]}
    dataset_name = "us_health_insurance_policy_corpus"
    dataset = Dataset.from_dict(data_dict)
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    #tokenizing the extracted data
    # 3. Set up the data loader and optimizer
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloader = DataLoader(tokenized_dataset, shuffle=True, batch_size=BATCH_SIZE_CONTINUED_TRAINING, collate_fn=data_collator)

    return train_dataloader

# Tokenize both the article and the summary
def tokenize_seq2seq(examples):
    # Tokenize the input text
    model_inputs = tokenizer(examples["article"], max_length=512, truncation=True)
    # Tokenize the target text and set it as 'labels'
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def load_training_data_for_multi_task_fine_tuning( unified_training_data, model ) :
    #new method
    data_dict = {'article': [sample['article'] for sample in unified_training_data],
                 "summary": [sample['summary'] for sample in unified_training_data]}
    dataset = Dataset.from_dict(data_dict)
    tokenized_dataset = dataset.map(tokenize_instruction_finetuning, batched=True, remove_columns=["article", "summary"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False )
    train_dataloader = DataLoader(tokenized_dataset, shuffle=True, batch_size=BATCH_SIZE_FINE_TUNING, collate_fn=data_collator)
    return train_dataloader

    #my method
    # data_dict = {'text': [strings  for strings in unified_training_data]}
    data_dict = {'text': unified_training_data}
    # data_dict = {'article': [ sample['article'] for sample in unified_training_data], "summary" : [sample['summary'] for sample in unified_training_data]}
    dataset_name = "article summary pairs"
    dataset = Dataset.from_dict(data_dict)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    #tokenizing the extracted data
    # 3. Set up the data loader and optimizer
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False )
    train_dataloader = DataLoader(tokenized_dataset, shuffle=True, batch_size=BATCH_SIZE_FINE_TUNING, collate_fn=data_collator)
    return train_dataloader


def train_model(model, train_dataloader ,epochs = 3, accumulation_steps = 10, batch_size = 4 ):
    #first need to check if model state dict is already present.
    # model.to(device)
    # 4. Set up the optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = epochs
    num_training_steps = num_epochs * len(train_dataloader)
    accumulation_steps = accumulation_steps
    # accelerator = Accelerator(
    #     gradient_accumulation_steps=accumulation_steps,  # Integrate gradient accumulation
    #     mixed_precision="fp16"  # Enable mixed precision
    # )
    # model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)


    if False :
        return
    else :
        # Training loop
        for epoch in range(num_epochs):
            print("epoch ", epoch)
            model.train()
            # The inner loop logic remains very similar
            for batch_idx, batch in enumerate(train_dataloader):
                # print("training on batch id : ", batch_idx, ", batch content : ", **batch )


                # without accelerator
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs= model(**batch)
                loss= outputs.loss
                loss = loss / accumulation_steps
                loss.backward()
                # Gradient accumulation logic
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            optimizer.step()
            optimizer.zero_grad()



                # with accelerator.accumulate(model):  # Context manager for gradient accumulation
                #     outputs = model(**batch)
                #     loss = outputs.loss
                #     accelerator.backward(loss)
                #
                #     if accelerator.sync_gradients:  # Only step when gradients are ready
                #         optimizer.step()
                #         optimizer.zero_grad()
        print("saved model weights")
        return model


def test_model_after_fine_tuning(model, text , task="summary"):
    final_prompt=""
    if task== "summary":
        final_prompt = f"""### Instruction:
        Summarize the following article:

        {text}
        """
        tokenized_final_prompt = tokenizer(
            final_prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        outputs = model.generate(
            **tokenized_final_prompt,
            min_new_tokens=30,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )

        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print('decoded output ' , decoded_output)
        return decoded_output

    elif task== "q&a":
        question = text

        final_prompt = f"""### Instruction:
        Answer the following question:

        {question}

        ### Response:
        """



    return

