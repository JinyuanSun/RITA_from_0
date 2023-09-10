import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from RITA_s.rita_configuration import RITAConfig
from RITA_s.rita_modeling import RITAModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("RITA_s")


config = RITAConfig(
    d_model=128,
    num_layers=6,
    num_heads=4
)

model = RITAModelForCausalLM(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"Model size: {model_size/1000**2:.1f}M parameters")
size = int(model_size/1000**2)
# exit()

data_files = {"train": "seq_train.txt", "valid": "seq_valid.txt"}
cutinase_dataset = load_dataset("text", data_files=data_files)
print(cutinase_dataset)

context_length = 128

def tokenize(element):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


tokenized_datasets = cutinase_dataset.map(
    tokenize, 
    batched=True, 
    remove_columns=["text"]
)

tokenizer.eos_token = "<EOS>"
tokenizer.add_special_tokens({'pad_token': '<PAD>'})
tokenizer.pad_token = tokenizer.eos_token

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

args = TrainingArguments(
    output_dir="seq-128",
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    evaluation_strategy="steps",
    eval_steps=200,
    logging_steps=100,
    gradient_accumulation_steps=4,
    num_train_epochs=20,
    weight_decay=0.1,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=200,
    fp16=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)

trainer.train()

trainer.save_model(f"seq-{size}M")