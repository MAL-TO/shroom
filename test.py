import os

os.environ['HF_HOME'] = '/data1/malto/cache'

from transformers import TrainingArguments
import evaluate
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer
import numpy as np


checkpoint = "microsoft/deberta-xlarge-mnli"
#checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
print(os.environ['HF_HOME'])