#!/usr/bin/env python3
# Commented out IPython magic to ensure Python compatibility.
# %env LC_ALL=C.UTF-8
# %env LANG=C.UTF-8
# %env TRANSFORMERS_CACHE=/content/cache
# %env HF_DATASETS_CACHE=/content/cache
# %env CUDA_LAUNCH_BLOCKING=1

# Commented out IPython magic to ensure Python compatibility.

# Commented out IPython magic to ensure Python compatibility.
# %env WANDB_PROJECT={project-name}


from datasets import load_dataset, load_metric

import pandas as pd
import numpy as np

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


from transformers import TrainingArguments



from torch.utils.data import Dataset, DataLoader
import torchaudio
abs_path_to_data = "/data/users/maqsood/main_exp/thesis/telugu_asr/te"
os.environ["HF_DATASETS_CACHE"] = '/data/users/maqsood/hf_cache'
os.environ['WANDB_API_KEY'] = '4974252525bb0812ae76d6ed03cfa6fb40c3fe3f'
os.environ['WANDB_DIR'] = '/data/users/maqsood/hf_cache'
os.environ['WANDB_CACHE_DIR'] = '/data/users/maqsood/hf_cache'
os.environ['WANDB_CONFIG_DIR'] = '/data/users/maqsood/hf_cache'
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

train_df = pd.read_csv(f"{abs_path_to_data}/train.tsv", sep="\t")
_train_df = train_df.copy()
total_records = len(train_df)
train_df["id"] = range(0, total_records)
print(f"Step 0: {len(train_df)}")

train_df["path"] = abs_path_to_data + "/clips/" + train_df["path"]
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

test_df = pd.read_csv(f"{abs_path_to_data}/test.tsv", sep="\t")

_test_df = test_df.copy()
total_records = len(test_df)
test_df["id"] = range(0, total_records)
print(f"Step 0: {len(test_df)}")

test_df["path"] = abs_path_to_data + "/clips/" + test_df["path"]
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



train_df.to_csv("/data/users/maqsood/main_exp/thesis/telugu_asr/train.csv", sep="\t", encoding="utf-8", index=False)
test_df.to_csv("/data/users/maqsood/main_exp/thesis/telugu_asr/test.csv", sep="\t", encoding="utf-8", index=False)

print(train_df.shape)
print(test_df.shape)

open_slr_train = load_dataset("csv", data_files={"train": "/data/users/maqsood/main_exp/thesis/telugu_asr/train.csv"}, delimiter="\t")["train"]
open_slr_test = load_dataset("csv", data_files={"test": "/data/users/maqsood/main_exp/thesis/telugu_asr/test.csv"}, delimiter="\t")["test"]

print(open_slr_train)
print(open_slr_test)

chars_to_ignore_regex = '[\,\?\.\!\-\_\;\:\"\“\%\‘\”\।\’\']'

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


with open('vocab.json', 'w') as vocab_file:
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
trainset.to_csv("/data/users/maqsood/main_exp/thesis/telugu_asr/dataset/train.csv", sep="\t")


testset = []

for item in tqdm(open_slr_test, position=0, total=len(open_slr_test)):
    features = open_slr_test.features
    data = {}
    for key in features:
        data[key] = item[key]
    
    testset.append(data)

testset = pd.DataFrame(testset)
testset.to_csv("/data/users/maqsood/main_exp/thesis/telugu_asr/dataset/test.csv", sep="\t")

trainset.head()

testset.head()

save_dir = "/data/users/maqsood/main_exp/thesis/telugu_asr/wav2vec2-large-xlsr-telugu"
# save_dir = "/content/wav2vec2-large-xlsr-turkish"
# !ls {save_dir}



last_checkpoint = None

if os.path.exists(save_dir):
    last_checkpoint = get_last_checkpoint(save_dir)
    
print(last_checkpoint if last_checkpoint else 0)


if not os.path.exists(save_dir):
    print("NotExist")
    tokenizer = Wav2Vec2CTCTokenizer("/data/users/maqsood/main_exp/thesis/telugu_asr/vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
else:
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(save_dir)


if not os.path.exists(save_dir):
    print("NotExist")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
else:
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(save_dir)



if not os.path.exists(save_dir):
    print("NotExist")
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
else:
    processor = Wav2Vec2Processor.from_pretrained(save_dir)

if not os.path.exists(save_dir):
    print("NotExist")
    processor.save_pretrained(save_dir)
    print("Saved!")

open_slr_train[0]



def speech_file_to_array_fn(file_path):
    speech_array, _ = torchaudio.load(file_path)

    speech_array = speech_array[0].numpy()
    speech_array = librosa.resample(np.asarray(speech_array), 48_000, 16_000)
    sampling_rate = 16_000

    return speech_array, sampling_rate






class OpenslrDataset(Dataset):

    def __init__(self, csv_file, root_dir, processor, column_names=None, sep="\t"):
        self.data = pd.read_csv(os.path.join(root_dir, csv_file), sep=sep)
        self.processor = processor
        self.column_names = column_names

    def __len__(self):
        return len(self.data)


    def speech_file_to_array_fn(self, batch):
        speech_array, sampling_rate = torchaudio.load(batch["path"])
        batch["speech"] = speech_array[0].numpy()
        batch["sampling_rate"] = sampling_rate
        batch["target_text"] = batch["text"]
        return batch

    
    def resample(self, batch):
        batch["speech"] = librosa.resample(np.asarray(batch["speech"]), 48_000, 16_000)
        batch["sampling_rate"] = 16_000
        return batch

    
    def prepare_dataset(self, batch, column_names=None):
        batch["input_values"] = self.processor(batch["speech"], sampling_rate=batch["sampling_rate"]).input_values[0].tolist()

        with self.processor.as_target_processor():
            batch["labels"] = self.processor(batch["target_text"]).input_ids

        if column_names and isinstance(column_names, list):
            batch = {name: batch[name] for name in column_names}
        
        return batch


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        batch = self.data.iloc[idx].copy()
        batch = batch.to_dict()
        batch = self.speech_file_to_array_fn(batch)
        batch = self.resample(batch)
        batch = self.prepare_dataset(batch, self.column_names)

        return batch

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

train_dataset = OpenslrDataset("train.csv", "/data/users/maqsood/main_exp/thesis/telugu_asr/dataset/", processor=processor, column_names=["input_values", "labels"])
test_dataset = OpenslrDataset("test.csv", "/data/users/maqsood/main_exp/thesis/telugu_asr/dataset/", processor=processor, column_names=["input_values", "labels"])

print(len(train_dataset))
print(len(test_dataset))

for batch in train_dataset:
    print(batch.keys())
    print(type(batch))
    # print(batch)
    break

wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53" if not last_checkpoint else last_checkpoint, 
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.1,
    mask_time_prob=0.05,
    layerdrop=0.1,
    gradient_checkpointing=True, 
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=processor.tokenizer.vocab_size
)

print(len(processor.tokenizer))
print(processor.tokenizer.vocab_size)

model.freeze_feature_extractor()


training_args = TrainingArguments(
    # output_dir="/content/gdrive/MyDrive/wav2vec2-large-xlsr-turkish-demo",
    # output_dir="./wav2vec2-large-xlsr-turkish-demo",
    output_dir=save_dir,
    group_by_length=True,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=150,
    fp16=False,
    save_steps=400, # Just for demo, change it
    eval_steps=400, # Just for demo, change it
    logging_steps=400, # Just for demo, change it
    learning_rate=3e-4,
    weight_decay=0.1,
    warmup_steps=500, # Just for demo, change it
    save_total_limit=2,
    dataloader_num_workers=4,
    seed = 42,
)


class CommonVoiceTrainer(Trainer):

    def _get_train_sampler(self):
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
            self.train_dataset, collections.abc.Sized
        ):
            return None 
        
        if self.args.world_size <= 1:
            return RandomSampler(self.train_dataset)
        elif self.args.parallel_mode == ParallelMode.TPU and not self.args.dataloader_drop_last:
            # Use a loop for TPUs when drop_last is False to have all batches have the same size.
            return DistributedSamplerWithLoop(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
            )
        else:
            return DistributedSampler(
                self.train_dataset, num_replicas=self.args.world_size, rank=self.args.process_index
            )
    
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def _get_eval_sampler(self, eval_dataset):
        if self.args.local_rank != -1:
            return SequentialDistributedSampler(eval_dataset)
        else:
            return SequentialSampler(eval_dataset)


    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

trainer = CommonVoiceTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=processor.feature_extractor,
)

if last_checkpoint:
    print(f"last_checkpoint: {last_checkpoint}")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    train_result = trainer.train()
metrics = train_result.metrics
max_train_samples = len(train_dataset)
metrics["train_samples"] = min(max_train_samples, len(train_dataset))

trainer.save_model()

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

