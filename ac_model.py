import torch
from transformers import AutoModelForSequenceClassification, AutoConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

MODEL_NAMES2 = {
    "base": "bert-base-cased",
    "legal": "nlpaueb/legal-bert-base-uncased",
    "large": "bert-large-cased",
    }

MODEL_NAMES = {
    "base": r"C:\Users\epimenov\OneDrive - Government of Saskatchewan\Documents\_Personal\Bert Models\bert-base-cased",
    "legal": r"C:\Users\epimenov\OneDrive - Government of Saskatchewan\Documents\_Personal\Bert Models\legal-bert-base-cased",
    "large": r"C:\Users\epimenov\OneDrive - Government of Saskatchewan\Documents\_Personal\Bert Models\bert-large-cased",
    }


class ArgumentClassModel:

    def __init__(self, mode="base", num_labels=2, drop_prob=0.1):

        # self.device = device
        self.mode = mode.lower()
        self.num_labels = num_labels
        self.drop_prob = drop_prob
        self.id2label = {0: "REB", 1: "REF"}
        self.label2id = {"REB": 0, "REF": 1}

        # BERT model name
        if self.mode not in MODEL_NAMES:
            raise ValueError(f"Incorrect model selection: {self.mode}")
        self.model_name = str(MODEL_NAMES[self.mode])

        # Config
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.config.hidden_dropout_prob = drop_prob
        self.config.attention_probs_dropout_prob = drop_prob
        self.config.num_labels = num_labels
        self.config.id2label = self.id2label
        self.config.label2id = self.label2id

        # Load model
        if self.mode in ("base", "legal"):
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                config=self.config,
                device_map="cuda"
            )
            model.dropout.p = drop_prob

        elif self.mode == "large":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                config=self.config,
                device_map={"":0},
                quantization_config=bnb_config
            )
            model.dropout.p = drop_prob
            peft_config = LoraConfig(
                task_type="SEQ_CLS",
                inference_mode=False,
                r=128,
                lora_alpha=128,
                lora_dropout=0.01,
                bias="none",
                target_modules=[
                    "attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense",
                    "intermediate.dense", "output.dense"
                ])
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, peft_config)
            model.config.use_cache = False
            model.enable_input_require_grads()
            model.train()

        self.model = model