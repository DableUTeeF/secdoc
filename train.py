import torch
from torch.utils.data import DataLoader
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch import nn
from dataset import TXTData
from transformers import AutoTokenizer, AutoModel
from models import Estimater, Encoder
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def criterion(outputs, quality, binaries):
    pq = outputs[:, :4]
    qloss = crossentropy(pq, quality)
    pb = torch.sigmoid(outputs[:, 4:])
    bloss = bceloss(pb, binaries.long().float())
    pa = ((pq.max(1)[1] == quality).float().mean() + ((pb > 0.5) == binaries).float().mean()) / 2
    return qloss + bloss, (pq.max(1)[1] == quality).float().mean(), ((pb > 0.5) == binaries).float().mean()


def collate_fn(batch):
    txt = []
    quality = []
    binaries = []
    for x in batch:
        txt.append(x[0])
        quality.append(x[1])
        binaries.append(x[2])
    txt = tokenizer(txt,
                    padding=True,
                    # max_length=max_target_length,
                    return_tensors="pt",
                    # truncation=True
                    )
    quality = torch.tensor(quality)
    binaries = torch.tensor(binaries)
    return txt, quality, binaries


if __name__ == '__main__':
    world_size = 1
    rank = 0
    epochs = 30
    initial_lr = 1e-3
    train_sampler = None
    val_sampler = None
    shuffle = True
    device = 'cuda'
    train_set = TXTData('/project/lt200203-aimedi/palm/sec_doc_sum/data/ocr/', '/project/lt200203-aimedi/palm/sec_doc_sum/data/docs/Training_dataset.xlsx')
    valid_set = TXTData('/project/lt200203-aimedi/palm/sec_doc_sum/data/ocr/', '/project/lt200203-aimedi/palm/sec_doc_sum/data/docs/testing_dataset.xlsx')
    if world_size > 1:
        shuffle = False
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    loader_train = DataLoader(train_set, batch_size=8, shuffle=shuffle, sampler=train_sampler, num_workers=0, collate_fn=collate_fn)
    loader_val = DataLoader(valid_set, batch_size=8, shuffle=False, sampler=val_sampler, num_workers=0, collate_fn=collate_fn)

    encoder = Encoder().to(device)
    for param in encoder.parameters():
        param.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained('/project/lt200203-aimedi/palm/huggingface/wangchanberta-base-att-spm-uncased')
    model = Estimater().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=2e-5)
    scheduler = CosineLRScheduler(
        optim,
        t_initial=epochs * len(loader_train),
        lr_min=initial_lr * 0.01,
        warmup_t=5 * len(loader_train),
        t_in_epochs=False,
        warmup_lr_init=initial_lr * 0.01
    )
    crossentropy = nn.CrossEntropyLoss()
    bceloss = nn.BCELoss()
    train_steps = 0
    val_steps = 0
    now = datetime.now()
    log_dir = os.path.join(
        './logs',
        now.strftime('%d-%b-%Y_%H:%M:%S')
    )
    writer = SummaryWriter(
        log_dir=log_dir
    )

    for e in range(epochs):
        model.train()
        for i, (txt, quality, binaries) in enumerate(loader_train):
            train_steps += 1
            txt = txt.to(device)
            quality = quality.to(device)
            binaries = binaries.to(device)
            outputs = encoder(txt)
            outputs = model(outputs)

            optim.zero_grad()
            loss, qa, ba = criterion(outputs, quality, binaries)
            loss.backward()
            optim.step()
            scheduler.step(train_steps)
            writer.add_scalar("Loss/train", loss, train_steps)
            writer.add_scalar("QA/train", qa, train_steps)
            writer.add_scalar("BA/train", ba, train_steps)
            writer.add_scalar("LR", torch.tensor(scheduler._get_lr(train_steps)), train_steps)
            writer.add_scalar("Epoch", e + i / len(loader_train), train_steps)
        model.eval()
        vals = []
        predicts = []
        qas = []
        bas = []
        with torch.no_grad():
            for i, (txt, quality, binaries) in enumerate(loader_val):
                val_steps += 1
                if input is None:
                    continue
                txt = txt.to(device)
                quality = quality.to(device)
                binaries = binaries.to(device)
                outputs = encoder(txt)
                outputs = model(outputs)

                loss, qa, ba = criterion(outputs, quality, binaries)
                writer.add_scalar("Loss/val", loss, val_steps)
                writer.add_scalar("QA/val", qa, val_steps)
                writer.add_scalar("BA/val", ba, val_steps)
                qas.append(qa)
                bas.append(ba)
        writer.add_scalar("Val_QA", torch.stack(qas).mean(), val_steps)
        writer.add_scalar("Val_BA", torch.stack(bas).mean(), val_steps)
