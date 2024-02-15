import os
os.environ['HF_HOME'] = '/data1/malto/cache'


import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
import numpy as np
from pathlib import Path
from datasets import load_dataset, concatenate_datasets, DatasetDict
import scipy
import pandas as pd

import torch
import random
import transformers
from functools import partial

def preprocess_function(examples, tokenizer): # not batched
    model_inputs = tokenizer(examples['hyp'], examples['tgt'] if examples['ref'] != 'src' else examples['src'], truncation=True, max_length=80)
    return model_inputs

def get_label(examples): # not batched
    return {"label" : 1 if examples['p(Hallucination)'] > 0.5 else 0}

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

def train_with_dataset(ds, num_epochs, model, ds_val_aware, tokenizer, data_collator, BATCH_SIZE, BASE_DIR, reduce_dataset = False):
    ds = ds.map(get_label).map(partial(preprocess_function, tokenizer = tokenizer)).remove_columns(['hyp', 'src', 'task', 'ref', 'tgt', 'model', 'labels', 'C-W', 'p(Hallucination)']).shuffle()
    #ds = ds.shard(num_shards=5, index=0) if reduce_dataset else ds
    training_args = TrainingArguments(
        output_dir=BASE_DIR / "checkpoint" / "sequential",
        learning_rate=1e-6,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        report_to="none",
        save_strategy="no",
        logging_steps=1,
        lr_scheduler_type="constant"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        eval_dataset=ds_val_aware.map(get_label).map(partial(preprocess_function, tokenizer = tokenizer)).remove_columns(['hyp', 'src', 'task', 'ref', 'tgt', 'model', 'labels', 'C-W', 'p(Hallucination)']),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer

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

def run_sequential_finetuning(n, use_mnli):
    BASE_DIR = Path("/data1/malto/shroom/")
    BATCH_SIZE = 6
    NUM_EPOCHS_SYN = 1
    NUM_EPOCHS_GPT = 2
    NUM_EPOCHS_TRUE = 3

    set_seed(n)

    BASE_DIR = Path("/data1/malto/shroom/")

    checkpoint = "microsoft/deberta-xlarge-mnli" if use_mnli else "microsoft/deberta-v2-xlarge"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    id2label = {0: "Not Hallucination", 1: "Hallucination"}
    label2id = {"Not Hallucination": 0, "Hallucination": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
    )

    syntetic_test_size_split = 0.000000001

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

    ds_syn = concatenate_datasets([ds_mt['train'], ds_dm['train'], ds_pg['train']])

    train_with_dataset(ds_syn, NUM_EPOCHS_SYN, model, ds_val_aware, tokenizer, data_collator, BATCH_SIZE, BASE_DIR)
    train_with_dataset(ds_gpt['train'], NUM_EPOCHS_GPT, model, ds_val_aware, tokenizer, data_collator, BATCH_SIZE, BASE_DIR)
    trainer = train_with_dataset(ds_val['train'], NUM_EPOCHS_TRUE, model, ds_val_aware, tokenizer, data_collator, BATCH_SIZE, BASE_DIR)

    path = "paper_results_mnli/" if use_mnli else "paper_results/"


    ds_val_aware = load_dataset("json", data_files=["val.model-aware-cla.json"])
    predictions, _, _ = trainer.predict(ds_val_aware['train'].map(partial(preprocess_function_test, tokenizer = tokenizer)))
    predictions = scipy.special.expit(predictions)
    predictions = predictions[:, 1] / predictions.sum(axis=1)
    df = pd.DataFrame(predictions, columns=["sequential"])
    df.to_csv(path + f"sequential_val{n}.csv", index=False)

    ds_test = load_dataset("json", data_files=[str(BASE_DIR / f"test.model-agnostic.json")])
    predictions, _, _ = trainer.predict(ds_test['train'].map(partial(preprocess_function_test, tokenizer = tokenizer)))
    preds = scipy.special.expit(predictions)
    preds = preds[:, 1] / preds.sum(axis=1)

    df = pd.DataFrame(preds, columns=["sequential"])
    df.to_csv(path + f"sequential_test{n}.csv", index=False)

if __name__ == "__main__":
    tests_to_run = 5
    use_mnli = True
    for i in range(tests_to_run):
        print(f"Running test {i}")
        run_sequential_finetuning(i, use_mnli)