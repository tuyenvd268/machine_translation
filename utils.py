from nltk.translate.bleu_score import sentence_bleu
from glob import glob
import numpy as np
import pandas as pd
import torch
import regex
import os
import re

def preprocess_text(text, single_quota = "<quota>"):
    text = text.replace("(", " ( ")
    text = text.replace(")", " ) ")

    text = regex.sub(r"(?<=[A-Za-z])'(?=[A-Za-z])", r" '", text)
    text = regex.sub(r"(?<=[\s\.\,\?\!\;\:])'(?=[A-Za-z]{2,})", r" ' ", text)

    text = regex.sub(r"(?<=[a-zA-Z])'(?=[\s\.\,\?\!\;\:])", r" ' ", text)
    text = regex.sub(r"(?<=[A-Za-z\.\,\?\!\;\:])'(?=[\s\.\,\?\!\;\:])", r" ' ", text)
    
    text = regex.sub(r"^'(?=[A-Za-z])", r" ' ", text)
    text = regex.sub(r"(?<=[a-zA-Z\s\.\,\?\!\;\:])'$", r" ' ", text)
    
    text = regex.sub(r"\.\.\.", " ... ", text)
    text = regex.sub(r'\"', ' " ', text)
    text = regex.sub(r'(?<!\.\.)\. ', ' . ', text)
    text = regex.sub(r'(?<!\.\.)\.$', ' . ', text)

    text = regex.sub(r'\}', ' } ', text)
    text = regex.sub(r'\{', ' { ', text)
    text = regex.sub(r'\,', ' , ', text)
    text = regex.sub(r'\?', ' ? ', text)
    text = regex.sub(r'\!', ' ! ', text)
    text = regex.sub(r'\-', ' - ', text)
    text = regex.sub(r'\;', ' ; ', text)
    text = regex.sub(r'(?<![\d]):(?![\d])', ' : ', text)
    text = regex.sub(r'(?<![\d])/(?![\d])', ' / ', text)
    text = regex.sub(r'\s+', ' ', text)

    # special cases
    text = text.replace(" 'a ", " ' a ")
    text = text.replace(" ' ", f' {single_quota} ')

    text = text.strip().strip('"').strip()
    return text

def read_txt(path, columns="source"):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    df = pd.DataFrame(lines, columns=[columns])
    print(df.head(5))
    
    return df

def load_data(data_dir, tokenizer=None, max_length=192):
    files = glob(f'{data_dir}/*.en')
    splits = [os.path.basename(file).split(".")[0] for file in files]
    splits = list(set(splits))
    
    print(f"##########################")
    data = []
    for split in splits:
        source_path = f'{data_dir}/{split}.en'
        print(f"###Load data from: {source_path}")
        source_df = read_txt(path=source_path, columns="source")
        source_df["source"] = source_df["source"].apply(lambda x: f'en: {x}')
        
        target_path = f'{data_dir}/{split}.vi'
        target_df = read_txt(path=target_path, columns="target")
        target_df["target"] = target_df["target"].apply(lambda x: f'vi: {x}')
        
        tmp_data = pd.concat([source_df, target_df], axis=1)
        
        data.append(tmp_data)
        
    data = pd.concat(data, axis=0)
    
    if tokenizer is not None:
        source_length = data.source.parallel_apply(lambda x: len(tokenizer.tokenize(x)))
        target_length = data.target.parallel_apply(lambda x: len(tokenizer.tokenize(x)))
        print("###Num sample (before filter): ", data.shape)
        data = data[(source_length < max_length) & (target_length < max_length)]
        print("###Num sample (after filter): ", data.shape)
        print(f"##########################")
        
    data.reset_index(inplace=True)
    return data

def save_ckpt(path, model, optimizer, bleu, epoch, step):
    state_dict = {
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "bleu": bleu,
        "epoch": epoch,
        "step": step
    }
    
    torch.save(state_dict, path)
    print(f"saved state dict to {path}")
    
def parse_batch(batch, tokenizer, device):
    input_ids = batch["source_ids"].cuda()
    attention_mask = batch['source_mask'].cuda()
    target_ids = batch["target_ids"].cuda()
    
    output_ids = target_ids.contiguous()
    lm_labels = target_ids.clone().detach()
    lm_labels[target_ids == tokenizer.pad_token_id] = -100
    
    return input_ids, attention_mask, output_ids, lm_labels

def post_process(sample):
    if sample.startswith("en:"):
        sample = sample.lstrip("en:")
    else:
        sample = sample.lstrip("vi:")

    sample = sample.replace('<quota>', " ")
    sample = re.sub(r"[\,\.\<\>\:;\"\'?\\/\!\~\+\-\@\=\$\%\^\&\*]", " ", sample)
    sample = re.sub(r"\s+", " ", sample)
    sample = sample.lower()
    return sample.strip().split()

def calculate_score(predictions, actuals):
    predictions = list(map(post_process, predictions))
    actuals = list(map(post_process, actuals))

    print("Predict: ", predictions[0:2])
    print("Label: ", actuals[0:2])
    
    bleu_scores = []
    for pred, label in zip(predictions, actuals):
        bleu_score = sentence_bleu([label,], pred, weights=(0.25, 0.25, 0.25, 0.25))
        
        bleu_scores.append(bleu_score)
        
    return np.mean(bleu_scores)
