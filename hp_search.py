""" (c) Evgeny Pimenov, 2025 """

""" Argument Classification: Hyperparameter search"""


from .ac_model import MODEL_NAMES
from .finetune import set_all_seeds, load_model, load_accuracy
from .data import load_dataset

import os
import json
from pathlib import Path
from fire import Fire
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding


def hyperparameter_search(mode: str = "base"):
    """
    Runs hyperparameter search for the specified model
    Args:
        mode (str): model name
    """
    set_all_seeds(555)

    model_folder_name = Path(MODEL_NAMES[mode]).name
    output_dir_mode_opt = Path(__file__).parent / "Models" / model_folder_name / "_HP-Search-Runs"

    tokenizer, train_dataset, val_dataset = load_dataset(mode)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print(f"Model selected: {mode.upper()}.")

    def model_init(trial=None):
            """
            Initializes the model for a trial, sets dropout 
            Args:
                trial: hp search trial
            Returns:
                model
            """
            if mode not in MODEL_NAMES:
                raise ValueError(f"Incorrect model selection: {mode}")

            if trial is not None:
                drop_prob = trial.suggest_float("dropout", 0.0, 0.2)
            else:
                drop_prob = 0.1 
           
            model = load_model(mode=mode, drop_prob=drop_prob)
            
            return model

    
    def hp_space(trial):
        """
        Defines the hyperparameter search space
        """
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-7, 1e-4),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
            "num_train_epochs": trial.suggest_int("num_train_epochs", low=3, high=4, step=1),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.2),
            "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [1, 2]),
        }

    def save_best_hyperparameters(best_trial, output_dir_mode_hp, mode):
        """
        Saves the best hyperparameters to a JSON file
        """
        filename = f"best_hyperparameters_{mode}.json"

        filepath = os.path.join(output_dir_mode_hp, filename)
        with open(filepath, 'w') as f:
            json.dump(best_trial.hyperparameters, f, indent=4)

    compute_metrics = load_accuracy()

    training_args = TrainingArguments(
        seed=555,
        output_dir=output_dir_mode_opt,
        per_device_eval_batch_size=16,
        eval_strategy="epoch", 
        save_strategy="no",          
        load_best_model_at_end=False,
        push_to_hub=False,
        metric_for_best_model="accuracy", 
        dataloader_num_workers=1,
        fp16=(mode == "large"),
        remove_unused_columns=(mode != "large"), 
        gradient_checkpointing=(mode == "large"),
        gradient_checkpointing_kwargs={'use_reentrant':(mode == "large")},
        label_names=['labels'] 
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna", # Using Optuna
        hp_space=hp_space,
        n_trials=100,
    )

    print("Best trial results:")
    for k, v in best_trial.hyperparameters.items():
        print(f"  • {k}: {v}")
    print(f"Best objective: {best_trial.objective:.4f}")

    save_best_hyperparameters(best_trial, output_dir_mode_hp=output_dir_mode_opt, mode=mode)


if __name__ == "__main__":

    Fire({"hp_search": hyperparameter_search})
