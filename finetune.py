
# Import modules
import os
import json
import pandas as pd

import torch
from datasets import Dataset
import transformers
from transformers import (
    BitsAndBytesConfig, LlamaTokenizer, LlamaForCausalLM
)
from peft import (
    PeftModel, prepare_model_for_kbit_training,
    LoraConfig, get_peft_model
)

from utils.train_utils import GenerationCallback


HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")

# Open the configuration file
with open("./config/training.json", "r", encoding="utf-8") as file:
    config = json.load(file)

# Read the dataframe used for LLM fine-tuning
df = pd.read_csv(os.path.join(config["datadir"], "processed_df.csv"))

# Set the base and peft model id
base_model_id = config["llm"]["base_model_id"]
peft_model_id = config["llm"]["peft_model_id"]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = LlamaTokenizer.from_pretrained(
    base_model_id,
    token=HUGGING_FACE_API_KEY)
model = LlamaForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    token=HUGGING_FACE_API_KEY
)

# TODO: at the moment loading and unloading the peft models is useless
peft_model = PeftModel.from_pretrained(
    model, peft_model_id, token=HUGGING_FACE_API_KEY)

# Prepare model for kbit training
original_model = peft_model.unload()
model.gradient_checkpointing_enable()
kbit_model = prepare_model_for_kbit_training(original_model)

# LORA configuration file
peft_config = LoraConfig(
    r=config["training"]["lora"]["r"],
    lora_alpha=config["training"]["lora"]["lora_alpha"],
    lora_dropout=config["training"]["lora"]["lora_dropout"],
    bias="none",
    task_type="CAUSAL_LM"
)
lora_model = get_peft_model(kbit_model, peft_config)
lora_model.print_trainable_parameters()

# Create dataset from pandas Dataframe
dataset = Dataset.from_pandas(df)
tokenizer.pad_token = tokenizer.eos_token
ds = dataset.map(
    lambda samples: tokenizer(
        samples["query"],
        truncation=True,
        padding=True,
        max_length=config["training"]["tokenizer_max_length"],
    ),
    batched=True
)

# Create an instance of your custom callback
callback = GenerationCallback()

# Train, make sure to adjust hyperparams
trainer = transformers.Trainer(
    model=lora_model,
    train_dataset=ds,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        warmup_steps=config["training"]["warmup_steps"],
        max_steps=config["training"]["max_steps"],
        learning_rate=config["training"]["learning_rate"],
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(
        tokenizer,
        mlm=False),
)

callback.trainer = trainer
callback.tokenizer = tokenizer
trainer.add_callback(callback)

peft_model.model.config.use_cache = False

# Training loop
trainer.train()

# Save the fine-tuned model
trainer.save_model(config["output_model_name"])
