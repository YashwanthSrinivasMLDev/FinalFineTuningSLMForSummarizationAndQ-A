import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq ,DefaultDataCollator , DataCollatorForLanguageModeling, BitsAndBytesConfig
from datasets import  Dataset
from torch.utils.data import DataLoader
import pypdf
from accelerate import Accelerator


model_state_dict_weights_path = "artifacts/model/model_weights.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE_CONTINUED_TRAINING = 4
# BATCH_SIZE_CONTINUED_TRAINING = 1
BATCH_SIZE_FINE_TUNING = 4
# BATCH_SIZE_FINE_TUNING = 1
# base_model_for_this_project = "sshleifer/distilbart-cnn-12-6"
base_model_for_this_project = "TinyLlama/TinyLlama_v1.1"
tokenizer = AutoTokenizer.from_pretrained(base_model_for_this_project)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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
    model.resize_token_embeddings(len(tokenizer))

    # if tokenizer.pad_token is None:
    #     # Add a new special token for padding
    #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #     # Resize the model's token embeddings to include the new pad token

    return model

def load_insurace_datasets():
    # extracted policy information from pdfs found in this link : https://www.cms.gov/marketplace/resources/fact-sheets-faqs
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

def tokenize_pre_process_function_for_summarization(examples):
    inputs = examples['article']
    targets = examples['summary']

    tokenized_input = tokenizer(
        inputs,
        max_length=1500,
        truncation=True,
        padding=True
    )


    tokenized_output = tokenizer.as_target_tokenizer(
    # tokenized_output = tokenizer(
        targets,
        # max_length=254,
        # truncation= True,

    )

    tokenized_input['labels']=tokenized_output['input_ids']
    return tokenized_input

def preprocess_function_2(examples):
    inputs = examples["article"]
    targets = examples["summary"]
    model_inputs = tokenizer(
        [f"Summarize: {inp}" for inp in inputs],
        text_target=targets,  # this creates labels aligned with the target
        max_length=1024,
        truncation=True
    )
    return model_inputs

def preprocess_function_3(examples):
    model_inputs = tokenizer(
        examples["article"],
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["summary"],
            max_length=128,
            truncation=True,
            padding="max_length"
        )["input_ids"]

    # Replace padding token id’s with -100 for labels
    labels = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_seq]
        for label_seq in labels
    ]

    model_inputs["labels"] = labels
    return model_inputs


def preprocess_function_llama_4(examples):
    # Tokenize inputs (articles with instruction)
    inputs = [
        f"Summarize the following:\n\n{article}\n\nSummary:"
        for article in examples["article"]
    ]
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    # Tokenize targets (summaries) separately
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["summary"],
            max_length=128,
            truncation=True,
            padding="max_length"
        )["input_ids"]

    # Replace pad tokens in labels with -100
    labels = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label_seq]
        for label_seq in labels
    ]

    model_inputs["labels"] = labels
    return model_inputs

def preprocess_causal_sft(examples, max_source_len=512, max_target_len=128, max_seq_len=512):
    """
    Build input_ids as [prompt_ids + target_ids]
    Build labels as [-100 ... -100] for prompt tokens + [target_ids]
    Then pad/truncate both to max_seq_len.
    """
    prompts = [
        f"Summarize the following:\n\n{art}\n\nSummary:"
        for art in examples["article"]
    ]
    targets = [t for t in examples["summary"]]

    # tokenize without adding special tokens, we’ll manage EOS explicitly for the target
    prompt_enc = tokenizer(
        prompts,
        add_special_tokens=False,
        truncation=True,
        max_length=max_source_len
    )
    target_enc = tokenizer(
        targets,
        add_special_tokens=False,
        truncation=True,
        max_length=max_target_len
    )

    input_ids_batch = []
    attention_masks_batch = []
    labels_batch = []

    eos_id = tokenizer.eos_token_id

    for p_ids, t_ids in zip(prompt_enc["input_ids"], target_enc["input_ids"]):
        # append EOS to target (very important for causal LM)
        t_ids = t_ids + [eos_id]

        # concatenate
        input_ids = p_ids + t_ids
        labels = ([-100] * len(p_ids)) + t_ids  # mask prompt, learn on target

        # truncate to max_seq_len (truncate both in the same way)
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]

        # build attention mask before padding
        attn = [1] * len(input_ids)

        # pad to max_seq_len
        pad_len = max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            attn = attn + [0] * pad_len
            labels = labels + [-100] * pad_len  # never learn on pad

        input_ids_batch.append(input_ids)
        attention_masks_batch.append(attn)
        labels_batch.append(labels)

    return {
        "input_ids": input_ids_batch,
        "attention_mask": attention_masks_batch,
        "labels": labels_batch,
    }


def load_training_data_for_multi_task_fine_tuning( unified_training_data, model ) :
    #new method
    data_dict = {'article': [sample['article'] for sample in unified_training_data],
                 "summary": [sample['summary'] for sample in unified_training_data]}
    dataset = Dataset.from_dict(data_dict)
    tokenized_dataset = dataset.map(preprocess_causal_sft,
                                    batched=True,
                                    remove_columns=["article", "summary"])
    data_collator = DefaultDataCollator(return_tensors="pt")
    # data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer ,
    #                                     model=model,  # pass model so it knows padding side
    #                                        padding=True,
    #                                        max_length=512
    #                                        )
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False )
    train_dataloader = DataLoader(tokenized_dataset, shuffle=True,
                                  batch_size=BATCH_SIZE_FINE_TUNING,
                                  collate_fn=data_collator)
    return train_dataloader


def train_model(model, train_dataloader ,epochs = 3, accumulation_steps = 10, batch_size = 4 ):
    #first need to check if model state dict is already present.
    # model.to(device)
    # 4. Set up the optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = epochs
    num_training_steps = num_epochs * len(train_dataloader)
    accumulation_steps = accumulation_steps
    accelerator = Accelerator(
        gradient_accumulation_steps=accumulation_steps,  # Integrate gradient accumulation
        mixed_precision="fp16"  # Enable mixed precision
    )
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    model.train()

    if False :
        return
    else :
        # Training loop
        for epoch in range(num_epochs):
            print("epoch ", epoch)
            # The inner loop logic remains very similar
            for batch_idx, batch in enumerate(train_dataloader):
                # print("training on batch id : ", batch_idx, ", batch content : ", **batch )


                # without accelerator
            #     batch = {k: v.to(device) for k, v in batch.items()}
            #     outputs= model(**batch)
            #     loss= outputs.loss
            #     loss = loss / accumulation_steps
            #     loss.backward()
            #     # Gradient accumulation logic
            #     if (batch_idx + 1) % accumulation_steps == 0:
            #         optimizer.step()
            #         optimizer.zero_grad()
            # optimizer.step()
            # optimizer.zero_grad()



                # with accelerator.accumulate(model):  # Context manager for gradient accumulation
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:  # Only step when gradients are ready
                    optimizer.step()
                    optimizer.zero_grad()
        print("saved model weights")
        return model


from accelerate import Accelerator
import torch


def test_model_after_fine_tuning(model, train_dataloader, epochs=3, accumulation_steps=10, batch_size=4):
    accelerator = Accelerator(
        gradient_accumulation_steps=accumulation_steps,
        mixed_precision="fp16"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    model.train()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, batch in enumerate(train_dataloader):

            with accelerator.accumulate(model):
                outputs = model(**batch)  # Ensure batch has input_ids, attention_mask, labels

                loss = outputs.loss
                if not loss.requires_grad:
                    raise ValueError(f"Loss does not require grad at step {batch_idx}. Check your inputs!")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

    accelerator.wait_for_everyone()
    accelerator.save_state("model_checkpoint")
    print("Model saved.")
    return model

# def test_model_after_fine_tuning(model, text , task="summary"):
#     final_prompt=""
#     if task== "summary":
#         final_prompt = f"""### Instruction:
#         Summarize the following article:
#
#         {text}
#
#         ###Summary:
#         """
#         tokenized_final_prompt = tokenizer(
#             final_prompt,
#             return_tensors='pt',
#             padding=True,
#             truncation=True,
#             # max_length=512
#             max_length=1500,
#         ).to(device)
#
#         outputs = model.generate(
#             **tokenized_final_prompt,
#             min_new_tokens=100,
#             max_new_tokens=300,
#             pad_token_id=tokenizer.eos_token_id,
#             # early_stopping=True,
#             # do_sample=True,
#             max_length=1800,
#             # top_k=50,
#             # top_p = 0.95,
#         )
#
#         decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         print('decoded output ' , decoded_output)
#         return decoded_output
#
#     elif task== "q&a":
#         question = text
#
#         final_prompt = f"""### Instruction:
#         Answer the following question:
#
#         {question}
#
#         ### Response:
#         """
#
#
#
#     return

