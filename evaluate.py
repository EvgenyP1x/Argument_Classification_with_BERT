""" 2025 """

""" Argument Classification: Model evaluation and testing"""


from .ac_model import MODEL_NAMES
from .data import load_dataset
from .finetune import load_accuracy

import torch
from pathlib import Path
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, Trainer
import tkinter as tk
from tkinter import messagebox
from fire import Fire

from peft import PeftModel

def check_model_accuracy (mode="base"):
    """
    Loads a trained model and calculates its accuracy on the validation dataset
    Args:
        mode (str): model name
    Returns:
        str: validation accuracy
    """
    if mode not in MODEL_NAMES:
        raise ValueError(f"Incorrect model selection: {mode}")

    model_folder_name = Path(MODEL_NAMES[mode]).name
    tokenizer_dir = Path(__file__).parent / "Models" / model_folder_name
    model_dir = Path(__file__).parent / "Models" / model_folder_name / "best_model"

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer_saved = AutoTokenizer.from_pretrained(tokenizer_dir)

    _, _, val_dataset = load_dataset(mode)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer_saved)

    compute_metrics = load_accuracy()

    trainer = Trainer(
        model=model,
        processing_class=tokenizer_saved,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    metrics = trainer.evaluate(eval_dataset=val_dataset)
    acc = metrics["eval_accuracy"]
    loss = metrics["eval_loss"]

    res = f"{mode.upper()} model —->   Accuracy: {acc:.4f}, Validation Loss: {loss:.4f}"

    return res


def test_model(mode="base"):
    """
    Launches a Tkinter GUI for interactive text classification
    Args:
        mode (str): model name
    """
    if mode not in MODEL_NAMES:
        raise ValueError(f"Incorrect model selection: {mode}")

    model_folder_name = Path(MODEL_NAMES[mode]).name
    tokenizer_dir = Path(__file__).parent / "Models" / model_folder_name
    model_dir = Path(__file__).parent / "Models" / model_folder_name / "best_model"

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    if mode == "large":
        model.config.id2label = {0: "REB", 1: "REF"}
        model.config.label2id = {"REB": 0, "REF": 1}
    model.eval()

    # Create the main Tkinter window
    root = tk.Tk()
    root.title("Argument Classification Model by Oxana Pimenova (c)")
    root.geometry("700x400")

    # Text entry label and input field
    tk.Label(root, text=f"REB/REF Type Argument Classification\n [{mode.upper()} mode] \n \n Enter text to classify:").pack(pady=5)
    entry = tk.Text(root, height=4, width=50)
    entry.pack(pady=5)

    # StringVar to store and display the result
    result_var = tk.StringVar()
    tk.Label(root, textvariable=result_var, fg="blue").pack(pady=5)

    def classify_text():
        """
        Retrieves user input, tokenizes it, runs inference, displays the predicted label
        """
        text = entry.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Input Required", "Please enter some text.")
            return

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_class_id = logits.argmax().item()
        label = model.config.id2label[predicted_class_id]
        result_var.set(f"Predicted label: {label}")

    # Classification button  
    tk.Button(root, text="Classify as REB or REF", command=classify_text).pack(pady=5)

    # Output label
    result_var = tk.StringVar()
    tk.Label(root, textvariable=result_var, fg="blue").pack(pady=5)

    # Event loop
    root.mainloop()


if __name__ == "__main__":

    Fire({"check": check_model_accuracy, "evaluate": test_model})
