import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
from class_rnn_5_final import *
from utils import *
import time
import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LinearLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import sys

def setup():
    dist.init_process_group(backend="nccl")

def prepare(rank, world_size, dataset, batch_size=2, pin_memory=False, num_workers=16):
    dataset = load_dataset(dataset,'train')
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    return dataloader

def cleanup():
    dist.destroy_process_group()

def printarray(arr):
    output = ''
    for i in range(6):
        output += '[' + str(i+1) + '] ' + f'{arr[i]:.5f} '
    return output

def main():
    rank = int(os.environ["LOCAL_RANK"])
    world_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    setup()
    dataset = 'imagenet'
    depth = 8
    dataloader = prepare(rank, world_size, dataset)
    model = cordsnet(dataset=dataset,depth=depth).to(rank)

    if dataset in ['fashionmnist','cifar10']:
        num_classes=10
        cutmix = v2.CutMix(num_classes=num_classes)
        mixup = v2.MixUp(num_classes=num_classes)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    if dataset in ['cifar100']:
        num_classes=100
        cutmix = v2.CutMix(num_classes=num_classes)
        mixup = v2.MixUp(num_classes=num_classes)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    
    model.load_state_dict(remove_data_parallel(torch.load('./results_' + str(depth) + '_' + dataset + '/4_parametric_20.pth')),strict=False)
    model = DDP(model, device_ids=[rank], output_device=rank)
    torch.save(model.state_dict(),'./results_' + str(depth) + '_' + dataset + '/5_finetune_0.pth')

    optimizer = optim.AdamW(model.parameters(),lr=0.00001)
    optimizer.zero_grad(set_to_none=True)
    
    log_size = 1000
    train_size, test_size = get_size(dataset)
    alpha = 0.2
    ramp =  torch.logspace(start=-3, end=0, steps=30).to(rank)

    for epoch in range(20):

        dataloader.sampler.set_epoch(epoch)  
        
        if rank == 0:
            start = time.time()
            batch_start = time.time() 
            batch_accs = np.zeros(6)
            batch_loss = np.zeros(6)
            batch_spon = 0.
            total_accs = np.zeros(6)
            total_loss = np.zeros(6)
            total_spon = 0.

        for i, data in enumerate(dataloader):

            if dataset in ['fashionmnist','cifar10','cifar100']:
                inputs, labels = cutmix_or_mixup(data)
            else:
                inputs, labels = data
            inputs, labels = inputs.to(rank), labels.to(rank)
            loss, accs, spon = model(inputs,labels,rank,alpha)
            full_loss = (loss*ramp).mean() + 1e-3*spon
            full_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.0001)
            optimizer.step()
            optimizer.zero_grad()

            # ----------------------------------------------------------
            torch.cuda.set_device(rank)
            accs_ = [None for _ in range(world_size)]
            loss_ = [None for _ in range(world_size)]
            spon_ = [None for _ in range(world_size)]
            dist.all_gather_object(accs_, accs)
            dist.all_gather_object(loss_, loss)
            dist.all_gather_object(spon_, spon)
            
            if world_rank == 0:
                devices = 4
                tempaccs = np.zeros([30])
                temploss = np.zeros([30])
                for j in range(devices):
                    tempaccs += accs_[j].cpu().detach().numpy()/devices
                    temploss += loss_[j].cpu().detach().numpy()/devices
                    total_spon += spon_[j].cpu().detach().numpy()/(train_size)
                    batch_spon += spon_[j].cpu().detach().numpy()/(devices*log_size)
                for j in range(6):
                    batch_accs[j] += np.mean(tempaccs[j*5:j*5+5])/log_size
                    batch_loss[j] += np.mean(temploss[j*5:j*5+5])/log_size
                    total_accs[j] += np.mean(tempaccs[j*5:j*5+5])*devices/train_size
                    total_loss[j] += np.mean(temploss[j*5:j*5+5])*devices/train_size

                if i%log_size == (log_size-1):
                    print('Epoch',str(epoch+1),'Batch',str(i+1),f'Time taken: {time.time() - batch_start:.5f}', flush=True)
                    print('Acc :',printarray(batch_accs), flush=True)
                    print('Loss:',printarray(batch_loss), flush=True)
                    torch.save(model.state_dict(),'./results_' + str(depth) + '_' + dataset + '/5_finetune_' + str(epoch+1) + '.pth')
                    batch_start = time.time() 
                    batch_accs = np.zeros(6)
                    batch_loss = np.zeros(6)
                    batch_spon = 0.

        if rank == 0:
            print('Epoch',str(epoch+1),f'Time taken: {time.time() - start:.5f}', flush=True)
            print('Acc :',printarray(total_accs), flush=True)
            print('Loss:',printarray(total_loss), flush=True)
            torch.save(model.state_dict(),'./results_' + str(depth) + '_' + dataset + '/5_finetune_' + str(epoch+1) + '.pth')

    cleanup()

if __name__ == '__main__':
    main()  
