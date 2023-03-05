import os
os.environ["HF_DATASETS_CACHE"] = '/data/users/maqsood/hf_cache'
from datasets import load_dataset, load_metric,concatenate_datasets,Dataset
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2ForPreTraining,Wav2Vec2Processor,Wav2Vec2FeatureExtractor,Wav2Vec2CTCTokenizer
from torch.utils.data import DataLoader
import torch
import numpy as np
import torchaudio
import pandas as pd
#import weights and bias
import wandb
import asyncio
import torch
#import Kmeans and pairwise_distances_argmin_min
from sklearn.cluster import KMeans
#import pairwise_distances
from sklearn.metrics.pairwise import pairwise_distances
import pickle
from datasets import disable_caching
disable_caching()
#set seed for reproducibility
#load data from pickle file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "pretrained"
dataset_path = "/data/corpora/openslr/telugu/"
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

async def pruning_function(centroids,hidden_states,after_prune_percent,cluster_labels):
    after_prune = []
    for lb,centroid in enumerate(centroids):
        cluster = np.where(cluster_labels == lb)
        distances = pairwise_distances(hidden_states[cluster], centroid.reshape(1, -1))
        threshold = np.percentile(distances, after_prune_percent)
        within_threshold = np.where(distances <= threshold)
        after_prune.extend(within_threshold[0])
    with open(f"/data/users/maqsood/main_exp/thesis/telugu_asr/sample_pruning/after_prune_{after_prune_percent}_telugu_task_adapted.pkl", "wb") as f:
        pickle.dump(after_prune, f)



df = pd.read_csv(f"{dataset_path}/processed/train.csv", sep="\t")
# df["path"] = f"{dataset_path}/processed/clips/" + df["path"]
test_dataset = Dataset.from_pandas(df)
model_checkpoint = "facebook/wav2vec2-large-xlsr-53"
model = Wav2Vec2ForPreTraining.from_pretrained(model_checkpoint)
model.config.output_hidden_states = True

model.load_state_dict(torch.load("/data/users/maqsood/thesis/pretrained/wav2vec2-large-xlsr-53telugu_asrpretraining_on_telugu_task_adaptation.pt", map_location=torch.device('cpu')), strict=False)
load_dataset_path = "/data/corpora/openslr/telugu/processed"
tokenizer = Wav2Vec2CTCTokenizer(f"{load_dataset_path}/vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
resampler = torchaudio.transforms.Resample(48_000, 16_000)
model.to("cuda")
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resampler(speech_array).squeeze()
    return processor(batch["speech"], sampling_rate=16_000, padding=True)
test_dataset = test_dataset.map(speech_file_to_array_fn)
test_loader = DataLoader(test_dataset, batch_size=1)
# Preprocessing the datasets.
# We need to read the aduio files as arrays
hidden_layer_append= torch.tensor([])
for batch in test_loader:
    
    with torch.no_grad():
        hidden_state = model(torch.tensor(batch["input_values"],dtype = torch.float).to("cuda"), attention_mask=torch.tensor(batch["attention_mask"],dtype = torch.float).to("cuda")).hidden_states[-1].to("cpu")
    
    hidden_state = hidden_state.mean(dim=1)
    hidden_state = hidden_state.reshape(hidden_state.shape[0], -1)
    hidden_layer_append = torch.cat((hidden_layer_append,hidden_state),0)
kmeans = KMeans(n_clusters=5, random_state=0).fit(hidden_layer_append)
#get centroids of the clusters
centroids = kmeans.cluster_centers_
cluster_labels = kmeans.predict(hidden_layer_append)  
after_prune_percentage = (10,)
loop = asyncio.get_event_loop()
loop.run_until_complete(asyncio.gather(*[pruning_function(centroids,hidden_layer_append,after_prune_percent,cluster_labels) for after_prune_percent in after_prune_percentage]))
loop.close()
