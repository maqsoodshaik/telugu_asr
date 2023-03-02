

from datasets import load_dataset, load_metric,Dataset
import wandb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import os
import string
import six
import re
import librosa

import json
from transformers.trainer_utils import get_last_checkpoint

import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from datasets import ClassLabel
import random
from transformers import Wav2Vec2ForCTC

import collections

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from torch.utils.data.sampler import RandomSampler, SequentialSampler
from transformers import Trainer
from transformers.trainer import (
    SequentialDistributedSampler, 
    SequentialSampler,
    DistributedSamplerWithLoop
)
from transformers.trainer import is_datasets_available


from transformers import TrainingArguments,Wav2Vec2Model,Wav2Vec2ForPreTraining



from torch.utils.data import DataLoader
import torchaudio
abs_path_to_data = "/data/users/maqsood/main_exp/thesis/telugu_asr/kannada/"
dataset_path = "/data/corpora/openslr/telugu/"
def normalizer(text):
    # Use your custom normalizer
    text = text.replace("\\n","\n")
    text = ' '.join(text.split())
    if re.search(r'''(&|[a-z]+)''',text,flags=re.IGNORECASE):
        return None
    text = re.sub(r'''%'''," శాతం ", text)
    text = re.sub(r'''(/|-|_)'''," ", text)
    text = re.sub("ై","ై", text)
    text = text.strip()
    return text

def normalizer_kannada(text):
    # Use your custom normalizer
    text = text.replace("\\n","\n")
    text = ' '.join(text.split())
    if re.search(r'''(&|[a-z]+)''',text,flags=re.IGNORECASE):
        return None
    if re.search(r'''[0-9]+''',text,flags=re.IGNORECASE):
        return None
    text = re.sub(r'''(/|-|_)'''," ", text)
    text = text.strip()
    return text

train_df = pd.read_csv(f"{dataset_path}/train.tsv", sep="\t")
_train_df = train_df.copy()
total_records = len(train_df)
train_df["id"] = range(0, total_records)
print(f"Step 0: {len(train_df)}")

train_df["path"] = dataset_path + "/clips/" + train_df["path"]
train_df["status"] = train_df["path"].apply(lambda path: True if os.path.exists(path) else None)
train_df = train_df.dropna(subset=["path"])
train_df = train_df.drop("status", 1)
print(f"Step 1: {len(train_df)}")

train_df["sentence"] = train_df["sentence"].apply(lambda t: normalizer(t))
train_df = train_df.dropna(subset=["sentence"])
print(f"Step 2: {len(train_df)}")

term_a = set(list(range(0, total_records)))
term_b = set(train_df["id"].values.tolist())
removed_items_train = [_train_df.iloc[index]["path"] for index in list(term_a - term_b)]
train_df = train_df.reset_index(drop=True)
train_df.head()

print(f"Items to be removed {len(removed_items_train)}")

test_df = pd.read_csv(f"{dataset_path}/test.tsv", sep="\t")

_test_df = test_df.copy()
total_records = len(test_df)
test_df["id"] = range(0, total_records)
print(f"Step 0: {len(test_df)}")

test_df["path"] = dataset_path + "/clips/" + test_df["path"]
test_df["status"] = test_df["path"].apply(lambda path: True if os.path.exists(path) else None)
test_df = test_df.dropna(subset=["path"])
test_df = test_df.drop("status", 1)
print(f"Step 1: {len(test_df)}")

test_df["sentence"] = test_df["sentence"].apply(lambda t: normalizer(t))
test_df = test_df.dropna(subset=["sentence"])
print(f"Step 2: {len(test_df)}")

term_a = set(list(range(0, total_records)))
term_b = set(test_df["id"].values.tolist())
removed_items_test = [_test_df.iloc[index]["path"] for index in list(term_a - term_b)]
test_df = test_df.reset_index(drop=True)
test_df.head()

print(f"Items to be removed {len(removed_items_test)}")

removed_items = removed_items_train + removed_items_test

for path in removed_items:
    if os.path.exists(path):
        os.remove(path)

text = " ".join(train_df["sentence"].values.tolist() + test_df["sentence"].values.tolist())
vocab = list(sorted(set(text)))

print(len(vocab), vocab)


# train_df.to_csv(f"{abs_path_to_data}/processed/train.csv", sep="\t", encoding="utf-8", index=False)
# test_df.to_csv(f"{abs_path_to_data}/processed/test.csv", sep="\t", encoding="utf-8", index=False)

# print(train_df.shape)
# print(test_df.shape)

# train_df = load_dataset("csv", data_files={"train":f"{abs_path_to_data}/processed/train.csv"}, delimiter="\t")["train"]
# test_df = load_dataset("csv", data_files={"test":f"{abs_path_to_data}/processed/test.csv"}, delimiter="\t")["test"]

# print(train_df)
# print(test_df)
#load from dataframes
open_slr_train =  Dataset.from_pandas(train_df)
open_slr_test =  Dataset.from_pandas(test_df)
chars_to_ignore_regex = '[\,\(\)\?\.\!\-\_\;\:\"\“\%\‘\”\।\’\'\u200d\u200c]'
def remove_special_characters(batch):
    text = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    text = normalizer(text)
    batch["text"] = text
    return batch

open_slr_train = open_slr_train.map(remove_special_characters, remove_columns=["sentence"])
open_slr_test = open_slr_test.map(remove_special_characters, remove_columns=["sentence"])

def extract_all_chars(batch):   
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = open_slr_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=open_slr_train.column_names)
vocab_test = open_slr_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=open_slr_test.column_names)

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(vocab_list)}
vocab_dict
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
len(vocab_dict)


with open(f"{dataset_path}/vocab.json", 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

# !mkdir -p /content/dataset

trainset = []

for item in tqdm(open_slr_train, position=0, total=len(open_slr_train)):
    features = open_slr_train.features
    data = {}
    for key in features:
        data[key] = item[key]
    
    trainset.append(data)

trainset = pd.DataFrame(trainset)
trainset.to_csv(f"{dataset_path}/processed/train.csv", sep="\t")


testset = []

for item in tqdm(open_slr_test, position=0, total=len(open_slr_test)):
    features = open_slr_test.features
    data = {}
    for key in features:
        data[key] = item[key]
    
    testset.append(data)

testset = pd.DataFrame(testset)
testset.to_csv(f"{dataset_path}/processed/test.csv", sep="\t")
