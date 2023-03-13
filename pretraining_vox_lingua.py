#!/usr/bin/env python3
import torch
from scipy.io.wavfile import write
import os
from torch.utils.data import DataLoader
from accelerate import Accelerator

os.environ["HF_DATASETS_CACHE"] = '/data/users/maqsood/hf_cache'
os.environ['WANDB_API_KEY'] = '4974252525bb0812ae76d6ed03cfa6fb40c3fe3f'
os.environ['WANDB_DIR'] = '/data/users/maqsood/hf_cache'
os.environ['WANDB_CACHE_DIR'] = '/data/users/maqsood/hf_cache'
os.environ['WANDB_CONFIG_DIR'] = '/data/users/maqsood/hf_cache'
from datasets import load_dataset
from transformers import AutoFeatureExtractor,Wav2Vec2ForPreTraining
from datasets import disable_caching
import wandb
from transformers import get_linear_schedule_with_warmup, AdamW,TrainingArguments
import sys
import numpy as np
# Add the directory containing mymodule to sys.path
sys.path.append('/data/users/maqsood/main_exp/thesis/telugu_asr/')
from data_collator_ctc import DataCollatorForWav2Vec2Pretraining
accelerator = Accelerator()
#set seed for reproducibility
def set_seed(seed):
    # Set the random seed manually for reproducibility.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
@torch.no_grad()
def eval_func(eval_dataloader, model,batch_size,len_eval):
    eval_loss = 0
    for batch in eval_dataloader:
        model.eval()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch["input_values"],mask_time_indices = batch['mask_time_indices'],sampled_negative_indices = batch['sampled_negative_indices'])
        loss = outputs.loss
        eval_loss += loss.item()
    eval_loss = eval_loss / len_eval
    wandb.log({f"Loss on validation of": eval_loss})
    return eval_loss
set_seed(42)
wandb.init(project='telugu_asr')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = wandb.config
config.dataset_name = "voxlingua"
config.epochs = 25
config.batch_size = 16
config.model = "voxlingua_telugu_pretraining"
disable_caching()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model_checkpoint = "facebook/wav2vec2-large-xlsr-53"
batch_size = 16
dataset_name = "voxlingua"
configs = ['te',]
for val,i in enumerate(configs):   
    dataset_train = load_dataset("/data/corpora/voxlingua/",data_dir= i)
    # dataset_train = dataset_train['train'].add_column("labels",[val]*len(dataset_train))
    dataset_train = dataset_train['train'].train_test_split(test_size=0.3)


feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
model = Wav2Vec2ForPreTraining.from_pretrained(model_checkpoint)
max_duration = 5.0  # seconds

"""We can then write the function that will preprocess our samples. We just feed them to the `feature_extractor` with the argument `truncation=True`, as well as the maximum sample length. This will ensure that very long inputs like the ones in the `_silence_` class can be safely batched."""
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * max_duration), 
        truncation=True,
        padding=True 
    )
    return inputs


encoded_dataset_train = dataset_train['train'].map(preprocess_function, remove_columns=["audio"], batched=True)
encoded_dataset_validation = dataset_train['test'].map(preprocess_function, remove_columns=["audio"], batched=True)
encoded_dataset_train.set_format("torch")
encoded_dataset_validation.set_format("torch")
data_collator = DataCollatorForWav2Vec2Pretraining(model=model, feature_extractor=feature_extractor,padding='max_length',)
training_args = TrainingArguments(
    output_dir = "/data/users/maqsood/main_exp/thesis/telugu_asr",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=int(config.batch_size)/4,
    gradient_checkpointing=True,
)

if training_args.gradient_checkpointing:
    model.gradient_checkpointing_enable()
train_dataloader = DataLoader(encoded_dataset_train, batch_size=training_args.per_device_train_batch_size,shuffle=True,drop_last=True,collate_fn=data_collator,)
eval_dataloader = DataLoader(encoded_dataset_validation, batch_size=training_args.per_device_train_batch_size,collate_fn=data_collator)
len_eval = len(eval_dataloader)

config.lr = 3e-5
optimizer = AdamW(model.parameters(), lr=config.lr)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader)*config.epochs*0.1, num_training_steps=len(train_dataloader)*config.epochs)
# model.to(device)

model, optimizer, train_dataloader,eval_dataloader = accelerator.prepare(model,optimizer, train_dataloader,eval_dataloader)

loss_comp = 10000000000
for epoch in range(1, config.epochs + 1):
    print(f"Epoch {epoch}")
    n_correct = 0
    n_samples = 0
    for batch_iter,batch in enumerate(train_dataloader):
        model.train()
        # batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch["input_values"],mask_time_indices = batch['mask_time_indices'],sampled_negative_indices = batch['sampled_negative_indices'])
        loss_self_supervised = outputs.loss
        loss = loss_self_supervised
        print(f"loss in batch {batch_iter+1} is {loss}")
        #log loss using wandb
        wandb.log({"self_supervised_loss":loss_self_supervised})
        # loss.backward()
        accelerator.backward(loss)
        if batch_iter % training_args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
    with torch.no_grad():
        ret_loss =eval_func(eval_dataloader, model,config.batch_size,len_eval)
        if ret_loss<=loss_comp:
            print("Saving model")
            loss_comp = ret_loss
            torch.save(model.state_dict(), f"/data/users/maqsood/thesis/pretrained/voxlingua_telugu_pretraining_25_epochs.pt")
