import os
from io import BytesIO
import argparse
import requests
import numpy as np
from tqdm import tqdm
from PIL import Image
import wandb

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers.optimization import get_linear_schedule_with_warmup
from datasets import load_dataset, Features, Value, Sequence

import sys
sys.path.append(os.path.join(sys.path[0], "dalle2-laion"))
from dalle2_laion.scripts import InferenceScript
from laion_data import get_wds_dataset


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment_name", type=str,
        help="Name of experiment"
    )

    parser.add_argument(
        "--resume", default=None, type=str,
        help='Model checkpoint directory'
    )
    parser.add_argument(
        "--resume_tokenizer", default=None, type=str,
        help='Tokenizer checkpoint directory'
    )
    parser.add_argument(
        "--models_dir", default="./models", type=str,
        help='Where to store downloaded model files'
    )
    parser.add_argument(
        "--train_data", default="pipe:aws s3 cp s3://s-mas/cc3m/{00000..00329}.tar -", type=str,
        help='Dataset to use'
    )
    parser.add_argument(
        "--valid_data", default="pipe:aws s3 cp s3://s-mas/cc3m/{00000..00329}.tar -", type=str,
        help='Dataset split'
    )
    
    parser.add_argument(
        "--epochs", default=10, type=int,
        help='Number of epochs'
    )
    parser.add_argument(
        "--lr", default=1e-4, type=float,
        help='Learning rate'
    )
    parser.add_argument(
        "--batch_size", default=64, type=int,
        help='Batch Size'
    )
    parser.add_argument(
        "--warm_steps", default=1000, type=int,
        help='Number fo warmup steps'
    )
    parser.add_argument(
        "--save_freq", default=10000, type=int,
        help="Number of batches between checkpointing"
    )

    parser.add_argument(
        "--workers", default=4, type=int,
        help='Number of workers'
    )
    parser.add_argument(
        "--world_size", default=1, type=int,
        help='Number of GPUs in environment'
    )

    parser.add_argument(
        "--save_dir", default="pretrained", type=str,
        help='Save directory for models and tokenizers'
    )
    parser.add_argument(
        "--wandb", default=False, action="store_true",
        help="Whether or not to log with wandb"
    )

    return parser

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    "Cleans up the distributed environment"
    dist.destroy_process_group()

def fetch_dataloader(dataset, batch_size):
    """
        Fetches dataset from source and creates a dataloader
        TODO: Return Validation data as well if argument is set
    """
    print(f"Attempting to load {dataset} ...")
    dataloader = None
    num_samples = 0
    if dataset == "laion/laion-coco":
        dataset_features = Features({
                        "URL": Value(dtype='string', id=None),
                        "TEXT": Value(dtype='string', id=None),
                        "top_caption": Value(dtype='string', id=None),
                        "all_captions": Sequence(feature=Value(dtype='string')),
                        "all_similarities": Sequence(feature=Value(dtype='string')),
                        "WIDTH": Value(dtype='int64', id=None),
                        "HEIGHT": Value(dtype='int64', id=None), 
                        "similarity": Value(dtype='float64', id=None),
                        "hash": Value(dtype='int64', id=None), 
                        "pwatermark": Value(dtype='float64', id=None),
                        "punsafe": Value(dtype='float64', id=None)
                    })
        dataset = load_dataset(dataset, features=dataset_features, split="train", streaming=True)
        dataset = dataset.remove_columns(["all_captions", "all_similarities", "WIDTH", "HEIGHT", "similarity", "hash", "pwatermark", "punsafe"]).shuffle()

        def collate_fn(batch):
            texts, imgs = [], []
            for obj in batch:
                texts.append(obj['TEXT'])

                response = requests.get(obj['URL'], headers={'User-Agent': 'Mozilla/5.0'}, stream=True)
                response.raise_for_status()
                imgs.append(Image.open(BytesIO(response.content)).convert("RGB"))
            
            return texts, imgs
        
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True if device == "cuda" else False, collate_fn=collate_fn)
        num_samples = 6e8
    elif dataset == "laion/laion400m_new":
        dataset_features = Features({
                        "NSFW": Value(dtype='string', id=None),
                        "similarity": Value(dtype='float64', id=None),
                        "LICENSE": Value(dtype='string', id=None),
                        "caption": Value(dtype='string', id=None), 
                        "url": Value(dtype='string', id=None),
                        "key": Value(dtype='int64', id=None),
                        "original_width": Value(dtype='int64', id=None),
                        "original_height": Value(dtype='int64', id=None) 
                    })
        dataset = load_dataset(dataset, features=dataset_features, split="train", streaming=True)
        dataset = dataset.remove_columns(["NSFW", "similarity", "LICENSE", "key", "original_width", "original_height"]).shuffle()

        def collate_fn(batch):
            texts, imgs = [], []
            for obj in batch:
                texts.append(obj['caption'])

                response = requests.get(obj['url'], stream=True)
                response.raise_for_status()
                imgs.append(Image.open(BytesIO(response.content)).convert("RGB"))
            
            return texts, imgs
        
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True if device == "cuda" else False, collate_fn=collate_fn)
        num_samples = dataloader.num_samples
    elif dataset == "pipe:aws s3 cp s3://s-mas/cc3m/{00000..00329}.tar -":
        num_samples = 2905954
        dataloader = get_wds_dataset(args, True, num_samples, epoch=0, floor=False, tokenizer=None)
    else:
        raise NotImplementedError

    print("Created dataloader")

    return dataloader, num_samples


def fetch_models(model_dir, prior_url, decoder_url, decoder_config_url):
    os.makedirs(model_dir, exist_ok=True)
    
    # Download decoder
    print("Downloading decoder and decoder config")
    decoder_path = os.path.join(model_dir, "decoder.pth")
    decoder_config_path = os.path.join(model_dir, "decoder_config.json")
    
    if not os.path.isfile(decoder_path): 
        decoder_file = requests.get(decoder_url, allow_redirects=True)
        with open(decoder_path, 'wb') as f:
            f.write(decoder_file.content)
    if not os.path.isfile(decoder_config_path): 
        decoder_config_file = requests.get(decoder_config_url, allow_redirects=True)
        with open(decoder_config_path, 'wb') as f:
            f.write(decoder_config_file.content)
    
    # Download prior
    print("Downloading prior and prior config")
    prior_path = os.path.join(model_dir, "prior.pth")
    
    if not os.path.isfile(prior_path):
        prior_file = requests.get(prior_url, allow_redirects=True)
        with open(prior_path, 'wb') as f:
            f.write(prior_file.content)

    return decoder_path, decoder_config_path, prior_path


def text_to_CLIP(texts, imgs, lang_model, tokenizer, diffusion_prior):
    batch_size = len(texts)

    criterion_ce = torch.nn.CrossEntropyLoss()
    criterion_mse = torch.nn.MSELoss()

    # Compute target embeddings
    with torch.no_grad():
        image_priors = diffusion_prior._sample_prior(texts, sample_count=1, batch_size=256, num_samples_per_batch=1)
        image_priors = torch.stack(list(map(lambda x: x[0], image_priors.values())), dim=0)

    # Predicting CLIP img embedding
    targets = tokenizer(texts, padding=True, return_tensors='pt', return_attention_mask=True)
    targets_ids = targets['input_ids'].to(device)
    targets_mask = targets['attention_mask'].to(device)

    token_embs = lang_model.transformer.wte(targets_ids)

    outputs = lang_model(
        inputs_embeds=token_embs,
        labels=targets_ids,
        return_dict=True,
        output_hidden_states=True,
        attention_mask=targets_mask
    )

    lm_loss = outputs['loss']

    # Find index of last token, our dummy token
    last_indices = targets_mask.sum(dim=1) - 1

    pred_priors = outputs['hidden_states'][-1][torch.arange(0, batch_size), last_indices]
    clip_loss = criterion_mse(pred_priors, image_priors)
    
    loss = lm_loss + clip_loss

    return loss


def train(args):
    """
    
    """
    print("Loading pretrained models ...")
    if args.resume_tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.add_special_tokens({"additional_special_tokens": ["[IMG OUT]"]})
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(args.resume_tokenizer)

    if args.resume is None:   
        lang_model = GPT2LMHeadModel.from_pretrained('gpt2')
    else:
        lang_model = GPT2LMHeadModel.from_pretrained(args.resume)

    if args.world_size > 1:
        raise NotImplementedError
        print("Setting up distributed training")
        setup(rank, args.world_size)
        rank = dist.get_rank()
        device_id = rank % torch.cuda.device_count()
        lang_model.to(device_id)
        lang_model = DDP(lang_model, device_ids=[device_id])
        print("Loading DALLE-2 models ...")

        inference = InferenceScript.create(os.path.join(sys.path[0], "dalle2-laion/configs/upsampler.example.json"), device=device_id)
    else:
        lang_model.to(device)
        print("Loading DALLE-2 models ...")
        inference = InferenceScript.create(os.path.join(sys.path[0], "dalle2-laion/configs/upsampler.example.json"), device=device)
    
    lang_model.train()
    # diffusion_prior.eval()
    # decoder.eval()

    print("Done loading models.")

    dataloader, num_samples = fetch_dataloader(args.train_data, args.batch_size)

    if args.train_data == "pipe:aws s3 cp s3://s-mas/cc3m/{00000..00329}.tar -":
        data = dataloader
    
    optimizer = torch.optim.AdamW(lang_model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warm_steps, num_training_steps=args.epochs * num_samples)


    for epoch in range(args.epochs):
       
        print(f"Training epoch {epoch}:")
        epoch_loss = 0.0

        if args.train_data == "pipe:aws s3 cp s3://s-mas/cc3m/{00000..00329}.tar -":
            data.set_epoch(epoch)
            dataloader = data.dataloader

        for batch_idx, (texts, imgs) in  tqdm(enumerate(dataloader), total=((num_samples-1)//args.batch_size)+1):

            optimizer.zero_grad()
            
            loss = text_to_CLIP(texts, imgs, lang_model, tokenizer, inference)
            loss.backward()

            optimizer.step()
            scheduler.step()

            if batch_idx % args.save_freq == 0 and batch_idx != 0:
                lang_model.save_pretrained(args.save_dir)
            
            if batch_idx % 1000 == 0 and batch_idx != 0:
                print(f"Loss at batch {batch_idx} = {loss}")
            
            if args.wandb:
                wandb.log({'loss': loss})

            epoch_loss += loss
        
        print(f"Loss at epoch {epoch+1}/{args.epochs} = {epoch_loss / (batch_idx+1)}")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device) 

    if args.wandb:
        wandb.init(
            project=args.experiment_name,
            
            config={
                "resume": args.resume,
                "dataset": args.train_data,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "warmup_steps": args.warm_steps
            }
        )

    train(args)

    if args.wandb:
        wandb.finish()