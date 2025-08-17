import json
import os

from TrainingSLM import load_insurace_datasets, create_model, train_model, \
    load_training_data_for_multi_task_fine_tuning
import torch
import traceback
from create_training_data_for_instruction_finetuning import create_training_data_summary_data_for_if, \
    create_training_data_qa_data_for_if
from peft import PeftModel, get_peft_model, LoraConfig, TaskType
from dotenv import load_dotenv
load_dotenv()

mode = os.getenv("mode")


if mode == "testing":
    print("in testing mode")
    EPOCH_CONTINUED_TRAINING = 1
    EPOCH_FINE_TUNING = 1
    ACCUMULATION_STEPS_FINETUNING = 1
    ACCUMULATION_STEPS_CONTINUED_TRAINING = 1

elif mode =="prod" :
    print("in production mode")
    EPOCH_CONTINUED_TRAINING = 3
    EPOCH_FINE_TUNING = 3
    ACCUMULATION_STEPS_CONTINUED_TRAINING = 16
    ACCUMULATION_STEPS_FINETUNING = 8

device = "cuda" if torch.cuda.is_available() else "cpu"
def start_main_app() :


    try :
        torch.cuda.empty_cache()
        train_dataloader_continued_training = load_insurace_datasets()
        print("dataloader : ", train_dataloader_continued_training)
        model_tiny_llama, tokenizer = create_model()

        if os.path.exists("./artifacts/model_continued_training"):
            print("model state dict for continued training already exists. loading it instead of retraining")
            model_trained_CT_on_insurance_datasets_tiny_llama = PeftModel.from_pretrained(model_tiny_llama, "./artifacts/model_continued_training" )
        else :
            #doing continued training on insurance datasets
            print('continued training on  insurance datasets')
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj"]  # Target more layers for better performance
            )
            model_tiny_lama_with_peft = get_peft_model(model_tiny_llama, peft_config)
            model_trained_CT_on_insurance_datasets_tiny_llama = train_model(model_tiny_lama_with_peft,
                                                                            train_dataloader_continued_training,
                                                                            epochs = EPOCH_CONTINUED_TRAINING,
                                                                            accumulation_steps= ACCUMULATION_STEPS_CONTINUED_TRAINING)
            print('saving model continued training state dict to disk')
            model_trained_CT_on_insurance_datasets_tiny_llama.save_pretrained("./artifacts/model_continued_training")


        #creating summary training data
        training_data_summaries = create_training_data_summary_data_for_if()
        training_data_qa = create_training_data_qa_data_for_if()
        consolidated_training_data_for_fine_tuning = training_data_summaries +  training_data_qa
        # consolidated_training_data_for_fine_tuning = training_data_summaries
        #creating dataloader for summary training data
        train_dataloader_summary_training_data = load_training_data_for_multi_task_fine_tuning(consolidated_training_data_for_fine_tuning, model_trained_CT_on_insurance_datasets_tiny_llama)
        #fiinetuning  the model on summary training data
        print('finetuning on summary + qa training data')
        if os.path.exists("artifacts/model_trained_FT_on_summary_tiny_llama"):
            print("model summary state dict exists. going to use that instead of retraining")
            model_trained_CT_on_insurance_datasets_tiny_llama.load_adapter("artifacts/model_trained_FT_on_summary_tiny_llama", adapter_name="summarization")
            model_trained_CT_on_insurance_datasets_tiny_llama.set_adapter("summarization")
            return model_trained_CT_on_insurance_datasets_tiny_llama, tokenizer
        else :
            print("model summary state dict doesn't exist. going to train the model from scratch")
            model_trained_FT_on_summary_tiny_llama = train_model(model_trained_CT_on_insurance_datasets_tiny_llama,
                                                                 train_dataloader_summary_training_data,
                                                                 epochs =EPOCH_FINE_TUNING ,
                                                                 accumulation_steps= ACCUMULATION_STEPS_FINETUNING)

            print("saving model summary state dict to disk")
            model_trained_FT_on_summary_tiny_llama.save_pretrained("artifacts/model_trained_FT_on_summary_tiny_llama")
            return model_trained_FT_on_summary_tiny_llama, tokenizer

    except Exception :
        print("exception : ",traceback.print_exc() )
        # print("cuda memory : ",torch.cuda.memory_summary())


