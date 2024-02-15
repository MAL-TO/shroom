import os
os.environ['HF_HOME'] = '/data1/malto/cache'

import evaluate
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset, concatenate_datasets, DatasetDict
import numpy as np
from datasets import load_dataset
from pathlib import Path

import random 
import torch 
import torch.nn as nn

from functools import partial


import scipy
import pandas as pd

def preprocess_function(examples, tokenizer): # not batched
    model_inputs = tokenizer(examples['hyp'], examples['tgt'] if examples['ref'] != 'src' else examples['src'], truncation=True, max_length=80)
    model_inputs["label"] = 1 if examples['p(Hallucination)'] > 0.5 else 0
    return model_inputs

def compute_metrics(eval_pred):
    #print(eval_pred)
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    #print(predictions, labels)
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def preprocess_function_test(examples, tokenizer): # not batched
    model_inputs = tokenizer(examples['hyp'], examples['tgt'], truncation=True, max_length=80)
    return model_inputs

def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    transformers.set_seed(random_seed)

    return torch.Generator().manual_seed(random_seed)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        p_hall = inputs.pop("p(Hallucination)")
        cond_weights = inputs.pop("C-W")
        #cond_weights = torch.where(cond_weights > 0.5, 1.1, 0.1)
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")[:, 1]
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        loss = cond_weights * loss_fn(logits, p_hall)
        loss = loss.mean()
        return (loss, outputs) if return_outputs else loss

def run_crlft(n, use_mnli):
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    os.environ["WANDB_DISABLED"] = "true"

    BATCH_SIZE = 6
    NUM_EPOCHS = 1
    BASE_DIR = Path("/data1/malto/shroom/")

    FREEZE = True
    FROZEN_LAYERS = 16
    USE_SEQUENTIAL = True

    checkpoint = "microsoft/deberta-xlarge-mnli" if use_mnli else "microsoft/deberta-v2-xlarge"
    #checkpoint = "microsoft/deberta-large-mnli"
    #checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    #checkpoint = "microsoft/deberta-v3-base"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    set_seed(n)
    
    id2label = {0: "Not Hallucination", 1: "Hallucination"}
    label2id = {"Not Hallucination": 0, "Hallucination": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
    )

    if USE_SEQUENTIAL:
        model.classifier = nn.Sequential(
            nn.Linear(in_features=1024 if use_mnli else model.deberta.encoder.conv.conv.weight.shape[1], out_features=2048, bias=True),
            nn.Sigmoid(),
            nn.Linear(in_features=2048, out_features=2, bias=True)
        )

    if FREEZE == True and checkpoint.startswith("microsoft"):
        print("freezing...")
        for param in model.deberta.embeddings.parameters():
            param.requires_grad = False
        for param in model.deberta.encoder.layer[:FROZEN_LAYERS].parameters():
            param.requires_grad = False
    
    
    syntetic_test_size_split = 0.01

    ds_mt = load_dataset("json", data_files=[str(BASE_DIR / f"train_labeled_MT_SOLAR.model-agnostic.json")])
    ds_dm = load_dataset("json", data_files=[str(BASE_DIR / f"train_labeled_DM_SOLAR.model-agnostic.json")])
    ds_pg = load_dataset("json", data_files=[str(BASE_DIR / f"train_labeled_PG_SOLAR.model-agnostic.json")])
    ds_val = load_dataset("json", data_files=[str(BASE_DIR / f"val.model-agnostic.json")])
    ds_val_aware = load_dataset("json", data_files=[str(BASE_DIR / f"val.model-aware.json")])
    ds_gpt = load_dataset("json", data_files=str(BASE_DIR / f"transformed_val_model_gpt.json"))

    ds_mt = ds_mt.remove_columns([el for el in ds_mt['train'].column_names if el not in ds_val['train'].column_names])['train'].train_test_split(test_size=syntetic_test_size_split)
    ds_dm = ds_dm.remove_columns([el for el in ds_dm['train'].column_names if el not in ds_val['train'].column_names])['train'].train_test_split(test_size=syntetic_test_size_split)
    ds_pg = ds_pg.remove_columns([el for el in ds_pg['train'].column_names if el not in ds_val['train'].column_names])['train'].train_test_split(test_size=syntetic_test_size_split)
    ds_gpt = ds_gpt.remove_columns([el for el in ds_pg['train'].column_names if el not in ds_val['train'].column_names])['train'].train_test_split(test_size=syntetic_test_size_split)


    ds = concatenate_datasets([ds_mt['train'], ds_dm['train'], ds_pg['train'], ds_val['train'], ds_gpt['train']])
    ds = ds.shuffle()
    ds = DatasetDict({
        'train' : ds,
        'test' : ds_val_aware['train'],
    })
    ds = ds.map(partial(preprocess_function, tokenizer = tokenizer))
    ds = ds.remove_columns(['hyp', 'src', 'task', 'ref', 'tgt', 'model', 'labels', 'label'])

    training_args = TrainingArguments(
        output_dir="/data1/malto/shroom/checkpoint/local_model",
        learning_rate=1e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        logging_steps=1,
        report_to="none",
        remove_unused_columns=False,
        lr_scheduler_type="constant"
    )

    trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            #compute_metrics=compute_metrics,
        )

    trainer.label_names = []
    trainer.can_return_loss = True

    trainer.train()

    path = "paper_results_mnli/" if use_mnli else "paper_results/"

    predictions, _, _ = trainer.predict(ds["test"])
    predictions = scipy.special.expit(predictions)
    predictions = predictions[:, 1] / predictions.sum(axis=1)
    df = pd.DataFrame(predictions, columns=["crlft"])
    df.to_csv(path+f"crlft_val{n}.csv", index=False)

    ds_test = load_dataset("json", data_files=["test.model-agnostic-cla.json"])
    obj = ds_test['train'].map(partial(preprocess_function_test, tokenizer = tokenizer)).remove_columns(["id", 'tgt', 'task', 'hyp', 'src'])

    predictions, _, _ = trainer.predict(obj)

    preds = scipy.special.expit(predictions)
    preds = preds[:, 1] / preds.sum(axis=1)

    df = pd.DataFrame(preds, columns=["crlft"])
    df.to_csv(path+f"crlft_test{n}.csv", index=False)


if __name__ == "__main__":
    tests_to_run = 5
    use_mnli = True
    for i in range(tests_to_run):
        print(f"Running test {i}")
        run_crlft(i, use_mnli)

    