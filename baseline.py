import os
os.environ['HF_HOME'] = '/data1/malto/cache'

import evaluate
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
import numpy as np
from datasets import load_dataset
from pathlib import Path

import random 
import torch 

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

def run_baseline(n, use_mnli):
    BASE_DIR = Path("/data1/malto/shroom/")
    BATCH_SIZE = 4
    NUM_EPOCHS = 10
    FREEZE = True
    FROZEN_LAYERS = 15

    set_seed(n)
    
    checkpoint = "microsoft/deberta-xlarge-mnli" if use_mnli else "microsoft/deberta-v2-xlarge"
    #checkpoint = "microsoft/deberta-v2-xxlarge-mnli"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    id2label = {0: "Not Hallucination", 1: "Hallucination"}
    label2id = {"Not Hallucination": 0, "Hallucination": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
    )

    if FREEZE == True and checkpoint.startswith("microsoft"):
        print("freezing...")
        for param in model.deberta.embeddings.parameters():
            param.requires_grad = False
        for param in model.deberta.encoder.layer[:FROZEN_LAYERS].parameters():
            param.requires_grad = False
    
    ds_val = load_dataset("json", data_files=[str(BASE_DIR / f"val.model-agnostic.json")]).map(partial(preprocess_function, tokenizer = tokenizer))
    ds_val_aware = load_dataset("json", data_files=[str(BASE_DIR / f"val.model-aware.json")]).map(partial(preprocess_function, tokenizer = tokenizer))
    ds_val = ds_val.remove_columns(['labels', 'model', 'ref', 'hyp', 'task', 'tgt', 'p(Hallucination)', 'src', 'C-W'])
    ds_val_aware = ds_val_aware.remove_columns(['labels', 'model', 'ref', 'hyp', 'task', 'tgt', 'p(Hallucination)', 'src', 'C-W'])

    training_args = TrainingArguments(
        output_dir=BASE_DIR / "checkpoint" / "sequential",
        learning_rate=1e-6,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
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
        train_dataset=ds_val['train'].shuffle(),
        eval_dataset=ds_val_aware['train'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    path = "paper_results_mnli/" if use_mnli else "paper_results/"

    predictions, _, _ = trainer.predict(ds_val_aware['train'])
    predictions = scipy.special.expit(predictions)
    predictions = predictions[:, 1] / predictions.sum(axis=1)
    df = pd.DataFrame(predictions, columns=["baseline"])
    df.to_csv(path+f"baseline_val{n}.csv", index=False)

    ds_test = load_dataset("json", data_files=[str(BASE_DIR / f"test.model-agnostic.json")])

    predictions, _, _ = trainer.predict(ds_test['train'].map(partial(preprocess_function_test, tokenizer = tokenizer)))

    preds = scipy.special.expit(predictions)
    preds = preds[:, 1] / preds.sum(axis=1)

    df = pd.DataFrame(preds, columns=["baseline"])
    df.to_csv(path+f"baseline_test{n}.csv", index=False)


if __name__ == "__main__":
    tests_to_run = 5
    use_mnli = True
    for i in range(tests_to_run):
        print(f"Running test {i}")
        run_baseline(i, use_mnli)

    