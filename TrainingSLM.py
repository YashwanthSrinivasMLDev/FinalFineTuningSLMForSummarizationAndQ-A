import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import pypdf
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from accelerate import Accelerator
#using automatic mixed precision to reduce  memory usage
from torch.amp import autocast,GradScaler

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
model_state_dict_weights_path = "./artifiacts/model/model_weights.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE_CONTINUED_TRAINING = 2
BATCH_SIZE_FINE_TUNING = 2


print("using device : ", device)

def tokenize_function(examples):
    # This function tokenizes the text data for the model
    return tokenizer(examples["text"], truncation=True, max_length=512)



def create_model():

    # using qlora peft for reduced memory usage
    # setting device

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj"]  # Target more layers for better performance
    )

    # Loading TinyLama from HuggingFace
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama_v1.1",
                                                 quantization_config=bnb_config,
                                                 device_map={"": 0})

    if tokenizer.pad_token is None:
        # Add a new special token for padding
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # Resize the model's token embeddings to include the new pad token
        model.resize_token_embeddings(len(tokenizer))

    model = get_peft_model(model, peft_config)
    # model.to(device)
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



def load_training_data_for_multi_task_fine_tuning( unified_training_data) :
    # data_dict = {'text': [strings  for strings in unified_training_data]}
    data_dict = {'text': unified_training_data}
    dataset_name = "article summary pairs"
    dataset = Dataset.from_dict(data_dict)
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    #tokenizing the extracted data
    # 3. Set up the data loader and optimizer
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
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
    accelerator = Accelerator(
        gradient_accumulation_steps=accumulation_steps,  # Integrate gradient accumulation
        mixed_precision="fp16"  # Enable mixed precision
    )
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    # if (os.path.exists(model_state_dict_weights_path)):
    #     print("found model weights. so, not retraining the model.")
    #     state_dict = torch.load(model_state_dict_weights_path, map_location=device)
    #     model.load_state_dict(state_dict)
    #     print("the loaded model from existing weights.pth : ", model )
    #     return model
    if False :
        return
    else :
        # Training loop
        for epoch in range(num_epochs):
            print("epoch ", epoch)
            model.train()
            # The inner loop logic remains very similar
            for batch_idx, batch in enumerate(train_dataloader):
                with accelerator.accumulate(model):  # Context manager for gradient accumulation
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:  # Only step when gradients are ready
                        optimizer.step()
                        optimizer.zero_grad()
        # saving the model
        # torch.save(model.state_dict(), model_state_dict_weights_path)
        print("saved model weights")
            # ... rest of your epoch loop
        return model


def test_model_after_fine_tuning(model, text , task="summary"):
    final_prompt=""
    if task== "summary":
        article_for_summary =  text

        final_prompt = f"""### Instruction:
        Summarize the following article:

        {article_for_summary}

        ### Response:
        """

    elif task== "q&a":
        question = text

        final_prompt = f"""### Instruction:
        Answer the following question:

        {question}

        ### Response:
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
        max_new_tokens=500,
        pad_token_id = tokenizer.eos_token_id
    )

    decoded_output = tokenizer.decode(outputs[0], skip_special_token=True)

    return decoded_output

