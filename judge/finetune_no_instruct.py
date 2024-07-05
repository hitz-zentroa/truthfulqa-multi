import gc
import os

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from trl import ORPOConfig, ORPOTrainer, setup_chat_format, SFTTrainer



def main():



    wandb.init(
    # set the wandb project where this run will be logged
    project="train_truthfulqa_judge",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 8e-6,
    "architecture": "llama",
    "dataset": "finetune_truth",
    "epochs": 5,
    }
    )

    if torch.cuda.get_device_capability()[0] >= 8:
        attn_implementation = "flash_attention_2"
        torch_dtype = torch.bfloat16
    else:
        attn_implementation = "eager"
        torch_dtype = torch.float16
    
    # Model
    base_model = 'meta-llama/Meta-Llama-3-8B'
    new_model = "Judge-Llama-3-8B-eng"

    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # LoRA config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
    tokenizer.add_eos_token = True
    #tokenizer.add_bos_token, tokenizer.add_eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
       base_model,
       quantization_config=bnb_config,
       device_map="auto",
       attn_implementation=attn_implementation
    )
    #model, tokenizer = setup_chat_format(model, tokenizer)
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False # silence the warnings. Please re-enable for inference!
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()

    dataset_name = "Blanca/truthful_judge" # TODO: upload to HF
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.shuffle(seed=42).select(range(1000))



    def format_chat_template(row):

        row["text"] = row['prompt'] + row['completion']
        return row


    dataset = dataset.map(
        format_chat_template,
        num_proc= os.cpu_count(),
    )
    dataset = dataset.train_test_split(test_size=0.01)

    # train the model

    training_args = TrainingArguments(
        output_dir="./results/",
        logging_strategy="epoch",  # Log at each training step
        evaluation_strategy="epoch",  # Evaluate at each training step
        save_strategy="epoch",  # Save at each training step
        learning_rate=0.1,
        per_device_train_batch_size=21,
        per_device_eval_batch_size=21,
        save_total_limit=3,
        load_best_model_at_end=True,
        num_train_epochs=5,
        lr_scheduler_type="inverse_sqrt",
        warmup_ratio=0.1,
        fp16=False,
        optim = "paged_adamw_8bit",
        report_to="wandb",
        run_name="trial",  # name of the W&B run (optional)
        logging_steps=1,
        #report_to='tensorboard',
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=peft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="text",
        max_seq_length=200,
        #callbacks = [EarlyStoppingCallback(early_stopping_patience=2)],
        packing=False
    )


    trainer.train()
    trainer.save_model(new_model)





if __name__ == '__main__':
    main()


    
