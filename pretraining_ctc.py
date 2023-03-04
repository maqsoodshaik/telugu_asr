#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
import numpy as np
from transformers import Wav2Vec2Config, Wav2Vec2Processor,Wav2Vec2ForPreTraining,AutoFeatureExtractor,get_linear_schedule_with_warmup, AdamW,Wav2Vec2CTCTokenizer,Wav2Vec2FeatureExtractor
from scipy.io.wavfile import write
import pandas as pd
from os import rename
from datasets import load_dataset, load_metric,concatenate_datasets,Audio
from torch.utils.data import Dataset
import wandb
import os
import sys
from sklearn.model_selection import train_test_split
import torchaudio
import librosa
# Add the directory containing mymodule to sys.path
sys.path.append('/data/users/maqsood/main_exp/thesis/telugu_asr/')
from datasets import disable_caching
disable_caching()
from data_collator_ctc import DataCollatorForWav2Vec2Pretraining
abs_path_to_data = "/data/users/maqsood/main_exp/thesis/telugu_asr/te"
os.environ["HF_DATASETS_CACHE"] = '/data/users/maqsood/hf_cache'
os.environ['WANDB_API_KEY'] = '4974252525bb0812ae76d6ed03cfa6fb40c3fe3f'
os.environ['WANDB_DIR'] = '/data/users/maqsood/hf_cache'
os.environ['WANDB_CACHE_DIR'] = '/data/users/maqsood/hf_cache'
os.environ['WANDB_CONFIG_DIR'] = '/data/users/maqsood/hf_cache'
class OpenslrDataset(Dataset):
    #create test and train splits
    def create_splits(self,train_split=0.9):
        train_dataset, test_dataset = train_test_split(self.data, test_size=1.0-train_split, random_state=42)
        return train_dataset, test_dataset

    def __init__(self, csv_file, root_dir, processor, column_names=None, sep="\t",split="train"):
        self.data = pd.read_csv(os.path.join(root_dir, csv_file), sep=sep)
        self.processor = processor
        self.column_names = column_names
        if split == "train":
            self.data , _ = self.create_splits(train_split=0.9)
        elif split == "validation":
            _, self.data = self.create_splits(train_split=0.9)
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
            # batch["labels"] = self.processor(batch["target_text"]).input_ids
            pass

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
load_dataset_path = "/data/corpora/openslr/telugu/processed"
tokenizer = Wav2Vec2CTCTokenizer(f"{load_dataset_path}/vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
#set seed for reproducibility
def set_seed(seed):
    # Set the random seed manually for reproducibility.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
#Evaluation function
#disable gradient calculation for the function
@torch.no_grad()
def eval_func(eval_dataloader, model,batch_size):
    eval_loss = 0
    for batch in eval_dataloader:
        model.eval()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch["input_values"],mask_time_indices = batch['mask_time_indices'],sampled_negative_indices = batch['sampled_negative_indices'])
        loss = outputs.loss
        eval_loss += loss.item()
    eval_loss = eval_loss / len(eval_dataloader)
    wandb.log({f"Loss on validation of": eval_loss})
    return eval_loss
set_seed(42)

#wandb initialization
wandb.init(project='telugu_asr')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = wandb.config
config.dataset_name = "telugu_asr"
config.epochs = 25
config.batch_size = 4
config.model = "pretraining_on_kannada_then_telugu_task_adaptation"
config.lr = 3e-5
#variables
model_checkpoint = "facebook/wav2vec2-large-xlsr-53"
model_name = model_checkpoint.split("/")[-1]+config.dataset_name+config.model
list_datasets_train = []
list_datasets_validation = []



# config_name = Wav2Vec2Config()
# config_name.output_hidden_states=True
# model = Wav2Vec2ForPreTraining(config=config_name)
model = Wav2Vec2ForPreTraining.from_pretrained(model_checkpoint)
#load model from state dict
model.load_state_dict(torch.load("/data/users/maqsood/thesis/pretrained/wav2vec2-large-xlsr-53telugu_asrpretraining_on_kannada_task_adaptation25_epochs0.065_mask_time_prob.pt",map_location = torch.device('cpu')))
feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint,return_attention_mask=True)
#loading dataset

dataset_train = OpenslrDataset("train.csv", f"{load_dataset_path}", processor=processor, column_names=["input_values"],split="train")
dataset_validation = OpenslrDataset("train.csv", f"{load_dataset_path}", processor=processor, column_names=["input_values"],split="validation")



# #preprocessing
# max_duration = 5.0  # seconds
# def preprocess_function(examples):
#     audio_arrays = [x["array"] for x in examples["audio"]]
#     inputs = feature_extractor(
#         audio_arrays, 
#         sampling_rate=feature_extractor.sampling_rate, 
#         max_length=int(feature_extractor.sampling_rate * max_duration), 
#         truncation="max_length",
#         padding="max_length" 
#     )
#     return inputs


# """To apply this function on all utterances in our dataset, we just use the `map` method of our `dataset` object we created earlier. This will apply the function on all the elements of all the splits in `dataset`, so our training, validation and testing data will be preprocessed in one single command."""

# encoded_dataset_train = dataset_train.map(preprocess_function, remove_columns=["audio","label"], batched=True)
# encoded_dataset_validation = dataset_validation.map(preprocess_function, remove_columns=["audio","label"], batched=True)
# encoded_dataset_train.set_format("torch")
# encoded_dataset_validation.set_format("torch")
data_collator = DataCollatorForWav2Vec2Pretraining(model=model, feature_extractor=feature_extractor,padding='max_length',)
train_dataloader = DataLoader(dataset_train, batch_size=config.batch_size,shuffle=True,drop_last=True,collate_fn=data_collator,)
eval_dataloader = DataLoader(dataset_validation, batch_size=config.batch_size,collate_fn=data_collator,)
#training
optimizer = AdamW(model.parameters(), lr=config.lr)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader)*config.epochs*0.1, num_training_steps=len(train_dataloader)*config.epochs)
model = model.to(device)


loss_comp = 10000000000
for epoch in range(1, config.epochs + 1):
    print(f"Epoch {epoch}")
    n_correct = 0
    n_samples = 0
    for batch_iter,batch in enumerate(train_dataloader):
        model.train()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch["input_values"],mask_time_indices = batch['mask_time_indices'],sampled_negative_indices = batch['sampled_negative_indices'])
        loss_self_supervised = outputs.loss
        loss = loss_self_supervised
        print(f"loss in batch {batch_iter+1} is {loss}")
        #log loss using wandb
        wandb.log({"self_supervised_loss":loss_self_supervised})
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    with torch.no_grad():
        ret_loss =eval_func(eval_dataloader, model,config.batch_size)
        if ret_loss<=loss_comp:
            print("Saving model")
            loss_comp = ret_loss
            torch.save(model.state_dict(), f"/data/users/maqsood/thesis/pretrained/{model_name}.pt")

