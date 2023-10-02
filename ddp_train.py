from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from datetime import datetime, timedelta
import torch.distributed as dist
from omegaconf import OmegaConf
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False, nb_workers=4)

from tqdm import tqdm
from glob import glob
import pandas as pd
import torch
import os
os.environ['TOKENIZERS_PARALLELISM'] = "false"
os.environ['OMP_NUM_THREADS'] = "1"

from dataset import EnViT5Dataset
from utils import (
    parse_batch,
    calculate_score,
    save_ckpt,
    load_data
)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed():
    dist_url = "env://"
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            timeout=timedelta(seconds=3600),
            rank=rank)

    torch.cuda.set_device(local_rank)
    dist.barrier()
    setup_for_distributed(rank == 0)

def init_dataset_and_dataloader(config, tokenizer):
    traindata = load_data(data_dir=config.train_dir, tokenizer=tokenizer)
    testdata = load_data(data_dir=config.test_dir, tokenizer=tokenizer)
    valdata = load_data(data_dir=config.val_dir, tokenizer=tokenizer)

    trainset = EnViT5Dataset(
        traindata, tokenizer=tokenizer, source_len=config.source_length, 
        target_len=config.target_len, source_text=config.source_text, target_text=config.target_text)
    testset = EnViT5Dataset(
        testdata, tokenizer=tokenizer, source_len=config.source_length, 
        target_len=config.target_len, source_text=config.source_text, target_text=config.target_text)
    valset = EnViT5Dataset(
        valdata, tokenizer=tokenizer, source_len=config.source_length, 
        target_len=config.target_len, source_text=config.source_text, target_text=config.target_text)
    
    sampler = DistributedSampler(dataset=trainset, shuffle=True)
    trainloader = DataLoader(
        dataset=trainset, batch_size=config.batch_size, sampler=sampler, shuffle=False, num_workers=config.n_worker)
    testloader = DataLoader(
        dataset=testset, batch_size=config.batch_size, shuffle=False, num_workers=config.n_worker)
    valloader = DataLoader(
        dataset=valset, batch_size=config.batch_size, shuffle=False, num_workers=config.n_worker)
    
    return trainloader, testloader, valloader

@torch.no_grad()
def validate(epoch, step, tokenizer, model, device, loader):
    model.eval()
    predictions, actuals = [], []
    
    print("###Start validate")
    bar = tqdm(loader, desc=f"Validate")
    for _, batch in enumerate(bar):
        output_ids = batch['target_ids'].to(device, dtype = torch.long)
        input_ids = batch['source_ids'].to(device, dtype = torch.long)
        attention_mask = batch['source_mask'].to(device, dtype = torch.long)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            generated_ids = model.module.generate(
                input_ids = input_ids,
                attention_mask = attention_mask, 
                max_length=config.max_length)
        
        preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        target = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        predictions.extend(preds)
        actuals.extend(target)
        
    model.train()
    
    bleu = calculate_score(predictions=predictions, actuals=actuals)
    return {
        "epoch":epoch, 
        "step": step,
        "bleu": bleu
    }
    
def train(config):
    init_distributed()

    print("###Init model and tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        "VietAI/envit5-translation", cache_dir="pretrained")  
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "VietAI/envit5-translation", cache_dir="pretrained").cuda()
    optimizer = torch.optim.Adam(params = model.parameters(), lr=config.lr)
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ['LOCAL_RANK'])])
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    print("###Init dataset and dataloader")
    trainloader, testloader, valloader = init_dataset_and_dataloader(
        config=config, tokenizer=tokenizer)
    step = 0
    
    print("###Start training")
    
    for epoch in range(config.n_epoch):
        model.train()
        bar = tqdm(trainloader, desc="Training")
        for index, batch in enumerate(bar):
            input_ids, attention_mask, output_ids, lm_labels = parse_batch(
                batch, tokenizer=tokenizer, device=config.device)
            # print(tokenizer.batch_decode(input_ids[0:2], skip_special_tokens=True))
            # print(tokenizer.batch_decode(lm_labels[0:2], skip_special_tokens=True))
            with torch.cuda.amp.autocast(dtype=torch.float16):
                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=lm_labels)

            scaler.scale(output.loss).backward()

            if (step + 1) % config.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                optimizer.zero_grad()
                scaler.update()

            step += 1
            bar.set_postfix(epoch=epoch, step=step, loss=output.loss.item())
            
            if is_main_process() and step % config.eval_and_save_per_step == 0:                
                result = validate(
                    epoch=epoch, tokenizer=tokenizer, step=step,
                    model=model, device=config.device, loader=testloader
                )
                print("\n###Test on Prep testset: ", result)
                path = f'{config.ckpt_dir}/ckpt_step={step}_epoch={epoch}_bleu={round(result["bleu"], 2)}.pt'
                
                save_ckpt(path=path, model=model, bleu=round(result["bleu"], 2),
                    epoch=epoch, step=step, optimizer=optimizer)
                
                result = validate(
                    epoch=epoch, tokenizer=tokenizer, step=step,
                    model=model, device=config.device, loader=valloader
                )
                print("\n###Test on PhoMT testset: ", result)

                
    print("###DONE!!")
    
if __name__ == "__main__":
    config = OmegaConf.load("config.yaml") 
      
    train(config)