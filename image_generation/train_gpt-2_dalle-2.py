import os
import argparse
import requests
import numpy
from tqdm import tqdm
from PIL import Image
import wandb

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers.optimization import get_linear_schedule_with_warmup
from datasets import load_dataset, Features, Value, Sequence
from dalle2_pytorch import DALLE2, DiffusionPriorNetwork, DiffusionPrior, Unet, Decoder, OpenAIClipAdapter, train_configs


device = "cuda" if torch.cuda.is_available() else "cpu"

def get_args_parser():
    parser = argparse.ArgumentParser()

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
        "--dataset", default="laion/laion-coco", type=str,
        help='Dataset to use'
    )
    parser.add_argument(
        "--dataset_split", default="train", type=str,
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
        "--save_dir", default="pretrained", type=str,
        help='Save directory for models and tokenizers'
    )
    parser.add_argument(
        "--wandb", default=False, action="store_true",
        help="Whether or not to log with wandb"
    )

    return parser


def fetch_dataloader(dataset, batch_size, split="train"):
    """
        Fetches dataset from source and creates a dataloader
    """
    print("Attempting to load {dataset} ...")
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
        dataset = load_dataset(dataset, features=dataset_features, split=split, streaming=True)
        dataset = dataset.remove_columns(["all_captions", "all_similarities", "WIDTH", "HEIGHT", "similarity", "hash", "pwatermark", "punsafe"]).shuffle()

        def collate_fn(batch):
            texts, imgs = [], []
            for obj in batch:
                texts.append(obj['TEXT'])
                imgs.append(Image.open(batch['URL']).convert("RGB"))
            
            return texts, imgs
        
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True if device == "cuda" else False, collate_fn=collate_fn)
        num_samples = 6e8
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
    
    decoder_file = requests.get(decoder_url, allow_redirects=True)
    with open(decoder_path) as f:
        f.write(decoder_file.content)
    decoder_config_file = requests.get(decoder_config_url, allow_redirects=True)
    with open(decoder_config_path) as f:
        f.write(decoder_config_file.content)
    
    # Download prior
    print("Downloading prior and prior config")
    prior_path = os.path.join(model_dir, "prior.pth")
    
    prior_file = requests.get(prior_url, allow_redirects=True)
    with open(prior_path) as f:
        f.write(prior_file.content)

    return decoder_path, decoder_config_path, prior_path


def text_to_CLIP(texts, imgs, lang_model, tokenizer, diffusion_prior):
    batch_size = len(texts)

    criterion_ce = torch.nn.CrossEntropyLoss()
    criterion_mse = torch.nn.MSELoss()

    CLIP_text_embs = diffusion_prior.clip.embed_image(texts).text_embed
    CLIP_img_embs = diffusion_prior.clip.embed_image(imgs).image_embed

    # Predicting CLIP img embedding
    targets = tokenizer(texts, padding=True, return_tensors='pt', return_attention_mask=True)
    targets_ids = targets['input_ids'].to(device)
    targets_mask = targets['attention_mask'].to(device)

    token_embs = lang_model.transformer.wte(targets_ids)

    dummy_token_emb = lang_model.transformer.wte(torch.tensor([0]).to(device)).reshape(1, -1)
    input_clip_embs = torch.cat((token_embs, dummy_token_emb, dummy_token_emb), dim=1)          # Add prediction CLIP text and image embeddings
    target_clip_ids = torch.cat((targets_ids, torch.zeros((batch_size, 2)).to(device)), dim=1)  # Add dummy tokens; is ignored in loss 
    target_clip_mask = torch.cat((targets_mask, torch.ones((batch_size, 2)).to(device)), dim=1) # Avoid masking new tokens

    outputs = lang_model(
        inputs_embeds=input_clip_embs,
        targets=target_clip_ids,
        return_dict=True,
        output_hidden_states=True,
        attention_mask=target_clip_mask
    )

    # Shift logits relative to target ids
    lm_loss = outputs['loss']

    pred_text_embs = outputs['hidden_states'][-1][:][-2]
    clip_text_loss = criterion_mse(pred_text_embs, CLIP_text_embs)

    pred_img_embs = outputs['hidden_states'][-1][:][-1]
    clip_img_loss = criterion_mse(pred_img_embs, CLIP_img_embs)

    # CLIP img embedding -> Image
    prior_loss = diffusion_prior(
                    text_embed = pred_text_embs,
                    image_embed = pred_img_embs
                )
    
    print(lm_loss, clip_text_loss, clip_img_loss, prior_loss)
    loss = lm_loss + clip_text_loss + clip_img_loss + prior_loss

    return loss


def train(args, num_samples):

    print("Loading pretrained models ...")
    if args.resume_tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(args.resume_tokenizer)

    if args.resume is None:   
        lang_model = GPT2LMHeadModel.from_pretrained('gpt2')
    else:
        lang_model = GPT2LMHeadModel.from_pretrained(args.resume)
    
    lang_model = lang_model.to(device)
    lang_model.train()

    prior_network = DiffusionPriorNetwork(
        dim=768,
        depth=24,
        dim_head=64,
        heads=32,
        normformer=True,
        attn_dropout=5e-2,
        ff_dropout=5e-2,
        num_time_embeds=1,
        num_image_embeds=1,
        num_text_embeds=1,
        num_timesteps=1000,
        ff_mult=4
    )
    diffusion_prior = DiffusionPrior(
        net = prior_network,
        clip = OpenAIClipAdapter("ViT-L/14"),
        timesteps = 100,
        cond_drop_prob = 0.2
    ).to(device)
    
    print("Loading DALLE-2 models ...")
    # Loading pre-trained diffusion_prior and decoder models
    # Freeze decoder and tune diffusion prior
    # Referencing https://github.com/LAION-AI/dalle2-laion/blob/main/notebooks/dalle2_laion_alpha.ipynb
    prior_url = "https://huggingface.co/zenglishuci/conditioned-prior/resolve/main/vit-l-14/prior_aes_finetune.pth"
    decoder_url = "https://huggingface.co/laion/DALLE2-PyTorch/resolve/main/decoder/1.5B_laion2B/latest.pth"
    decoder_config_url = "https://huggingface.co/laion/DALLE2-PyTorch/raw/main/decoder/1.5B/decoder_config.json"

    decoder_path, decoder_config_path, prior_path = fetch_models(args.models_dir, prior_url, decoder_url, decoder_config_url)
    prior_state_dict = torch.load(prior_path, map_location='device')
    diffusion_prior.load_state_dict(prior_state_dict['model'], strict=False)
    del prior_state_dict
    diffusion_prior.train()

    config = train_configs.TrainDecoderConfig.from_json_path(decoder_config_path)
    config.decoder.clip = None
    decoder = config.decoder.create().to(device)
    decoder_state_dict = torch.load(decoder_path, map_location='cpu')
    decoder.load_state_dict(decoder_state_dict, strict=False)
    del decoder_state_dict
    decoder.eval()

    print("Done loading models.")

    optimizer = torch.optim.AdamW(lang_model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warm_steps, num_training_steps=args.epochs * num_samples)

    for epoch in range(args.epochs):
       
        print(f"Training epoch: {epoch} -> ", end="")
        epoch_loss = 0.0

        for batch_idx, (texts, imgs) in  tqdm(enumerate(dataloader)):

            optimizer.zero_grad()
            
            loss = text_to_CLIP(texts, imgs, lang_model, tokenizer, diffusion_prior)
            loss.backward()

            optimizer.step()
            scheduler.step()
            
            if args.wandb:
                wandb.log({'loss': loss})

            epoch_loss += loss
        
        print(f"Loss at epoch {epoch+1}/{args.epochs} = {epoch_loss / (batch_idx+1)}")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    dataloader, num_samples = fetch_dataloader(args.dataset, args.batch_size, args.dataset_split)

    if args.wandb:
        wandb.init(
            project="general-gpt_gpt2-dalle-2",
            
            config={
                "resume": args.resume,
                "dataset": args.dataset,
                "data_split": args.dataset_split,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "warmup_steps": args.warm_steps
            }
        )

    train(args, num_samples)

    if args.wandb:
        wandb.finish()