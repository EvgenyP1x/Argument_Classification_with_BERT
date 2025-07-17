
from .ac_model import ArgumentClassModel, MODEL_NAMES
from .data import load_dataset

import os
import warnings
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from transformers import EarlyStoppingCallback
from transformers import set_seed, enable_full_determinism
from fire import Fire
from pathlib import Path
import numpy as np
import torch
import random
import json
import evaluate
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")


# SET SEED 
def set_all_seeds(seed: int = 555):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    enable_full_determinism(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)


# INITIATE ACCURACY
def load_accuracy():

    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    return compute_metrics


# LOAD MODEL
def load_model(mode = "base", num_labels: int = 2, drop_prob = 0.1):
    
    ac_model = ArgumentClassModel(mode, num_labels, drop_prob)
    return ac_model.model


# LOAD HYPERPARAMETERS
def load_model_config(mode: str):

    config_path = Path(__file__).parent / "Models" / "hyperparameters_current.json"
    with open(config_path, "r") as f:
        all_configs = json.load(f)
    return all_configs[mode.lower()]


# TRAIN
def train_model(mode: str = "base", hp = False):

    set_all_seeds(555)
    mode = mode.lower()

    model_folder_name = Path(MODEL_NAMES[mode]).name
    output_dir_model = Path(__file__).parent / "Models" / model_folder_name
    output_dir_model.mkdir(parents=True, exist_ok=True)    

    tokenizer, train_dataset, val_dataset = load_dataset(mode)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print(f"Model selected: {mode.upper()}.")

    if hp:
        try:
            print("Hyperparameters loaded.")
            hp_cfg = load_model_config(mode)
            lr = hp_cfg["learning_rate"]
            per_device_train_batch_size = hp_cfg["per_device_train_batch_size"]
            num_train_epochs = hp_cfg["num_train_epochs"]
            weight_decay = hp_cfg["weight_decay"]
            gradient_accumulation_steps = hp_cfg["gradient_accumulation_steps"]
            dropout = hp_cfg["dropout"]
        except KeyError as e:
            print(f"[ERROR] Mode '{mode}' not found in the config json.")
            raise SystemExit(1)  # stop execution cleanly
    else:
        print("Define hyperparameters manually in the function or use the default values as per below.")
        lr = 1e-05
        per_device_train_batch_size = 16
        num_train_epochs = 3
        weight_decay = 0.1
        gradient_accumulation_steps = 1
        dropout = 0.1

    print(f"Model Hyperparameters for mode: {mode}")
    print("-" * 40)
    print(f"Learning rate:                {lr}")
    print(f"Batch size:                   {per_device_train_batch_size}")
    print(f"Epochs:                       {num_train_epochs}")
    print(f"Weight decay:                 {weight_decay}")
    print(f"Gradient accumulation steps:  {gradient_accumulation_steps}")
    print(f"Dropout:                      {dropout}")

    def model_init():
        drop_prob = dropout

        if mode not in MODEL_NAMES:
            raise ValueError(f"Incorrect model selection: {mode}")
        
        model = load_model(mode=mode, drop_prob=drop_prob)

        # Checks model/tokenizer compatibility  
        if tokenizer.vocab_size != model.config.vocab_size:
            raise ValueError(
                f"Tokenizer vocab size ({tokenizer.vocab_size}) "
                f"does not match model config vocab size ({model.config.vocab_size})"
            )

        return model

    compute_metrics = load_accuracy()

    training_args = TrainingArguments(
        seed=555,
        output_dir=output_dir_model,
        learning_rate=lr, 
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
        push_to_hub=False,
        metric_for_best_model="accuracy", 
        dataloader_num_workers=1,
        fp16=(mode == "large"), # Enable mixed precision training
        remove_unused_columns=(mode != "large"), # LoRA par 
        gradient_checkpointing=(mode == "large"), # LoRA par
        gradient_checkpointing_kwargs={'use_reentrant':(mode == "large")}, # LoRA par
        label_names=['labels']
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience = 4)]
    )

    trainer.train()
    trainer.save_model(output_dir_model)
    tokenizer.save_pretrained(output_dir_model)


if __name__ == "__main__":

    Fire({"train": train_model})