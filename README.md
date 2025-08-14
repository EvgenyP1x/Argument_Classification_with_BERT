# BERT Finetuning for Argument Classification in a Government Context

## Project Overview

The project focuses on developing a classifier to identify argument types used in government negotiations with Indigenous groups and non-profit organizations. Three pre-trained BERT models were fine-tuned for this task: BERT Base (cased), BERT Large (cased) and BERT Legal Base (uncased).

## Requirements

Python 3.x

Libraries:
- transformers
- torch
- fire
- evaluate
- numpy
- optuna
- pandas
- scikit-learn
- peft

## Usage

The keyword **`base`** in the commands below can be replaced with **`legal`** or **`large`** to train the Legal or Large model, respectively.

To Clone repo:
```bash
git clone https://github.com/EvgenyP1x/Argument_Classification_with_BERT
```

To display information about the dataset:
```bash
python -m Argument_Classification_with_BERT.data test_dataset
```

To train the Bag-of-Words baseline model:
```bash
python -m Argument_Classification_with_BERT.bow_baseline baseline_train
```

To perform a hyperparameter search for the Base model:
```bash
python -m Argument_Classification_with_BERT.hp_search hp_search base
```

To train the Base model with the best hyperparameters:
```bash
python -m Argument_Classification_with_BERT.finetune train base -hp
```

To train the Base model with the default hyperparameters:
```bash
python -m Argument_Classification_with_BERT.finetune train base
```

To perform a hyperparameter search for the Base model:
```bash
python -m Argument_Classification_with_BERT.hp_search hp_search base
```

To evaluate the accuracy and loss of the trained Base model:
```bash
python -m Argument_Classification_with_BERT.evaluate check base 
```

To run the trained Base model for inference on new examples through the GUI:
```bash
python -m Argument_Classification_with_BERT.evaluate evaluate base
```

## Dataset

The dataset consists of 1,178 human-annotated examples, evenly split between the two classes, forming a balanced classification problem. Examples are selected and classified based on theoretical frameworks from document analysis and argumentation theory, and tokenized using the tokenizer of each respective model.

## Model Parameters

Three transformer models with varying sizes and architectures were finetuned: BERT Base cased (109M parameters, 12 transformer blocks, hidden size 768, 12 attention heads), BERT Large cased (335M parameters, 24 transformer blocks, hidden size 1024, 16 attention heads), Legal Base uncased (110M parameters, 12 transformer blocks, hidden size 768, 12 attention heads). A linear classification head with output size 2 was added on top, taking the final hidden state of the [CLS] token as input to predict the two target argument classes (REF and REB).

## Training Process

Hyperparameter tuning was performed using the Optuna framework, exploring learning rates, batch sizes, dropout, weight decay, gradient accumulation and number of training epochs. The best-performing hyperparameters were saved to **`Models/hyperparameters_current.json`** and could be loaded for finetuning.

The BERT Large model was finetuned using Low-Rank Adaptation and mixed precision to handle memory constraints. Given the small dataset (~1,000 examples), Base and Legal Base models reached optimal performance within 2–3 epochs, while the Large model required 1-2 additional epochs. Dropout, early stopping and weight decay were applied to prevent overfitting, and validation metrics were monitored to ensure generalization.

To establish a baseline, a Bag-of-Words approach was applied using combinations of classifiers and vectorizers.

## Performance and Analysis

All three fine-tuned BERT models outperformed the Bag-of-Words baseline, achieving classification accuracies in the range of 91–92%, compared to 85.17% for the benchmark. Among the BERT models, Base cased (91.95%) slightly outperformed Large cased (91.53%) and Legal Base uncased (91.53%), though differences were minimal and likely due to randomness in hyperparameter search.

The trained models were saved to the **`Models`** directory and can be used for inference via a tkinter-based GUI.

## Future Work

Despite the results, the small dataset raises concerns about generalization, as larger models are  prone to overfitting on limited data. 

Future improvements must involve expanding the dataset, applying data augmentation, exploring more domain-relevant pretraining and experimenting with alternative architectures, embeddings and training strategies.

## Conclusion

The project demonstrates that the finetuned language models pretrained with bidirectional masked language modeling are capable of performing well in tasks involving the detection of fallacious reasoning in institutional discourse. Project results confirm the potential of transformer-based models to capture linguistic patterns essential for robust argument classification. Ensuring the trained model can generalize well and is suitable for application in the real-life settings will require further development.

## License 

The project is licensed under the MIT License.
