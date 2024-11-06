import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
from class_rnn_2_inverse import *
from utils import *
import time
import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LinearLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import sys

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def prepare(rank, world_size, dataset, batch_size=64, pin_memory=False, num_workers=16):
    dataset = load_dataset(dataset,'train')
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    return dataloader

def cleanup():
    dist.destroy_process_group()

def printarray(arr):
    output = ''
    size = len(arr)
    for i in range(size):
        output += '[' + str(i+1) + '] ' + f'{arr[i]:.5f} '
    return output

def main(rank, world_size):
    setup(rank, world_size)
    dataset = 'cifar10'
    depth = 2
    areas = range(depth)
    dataloader = prepare(rank, world_size, dataset)
    model = step2cordsnetinverse(dataset=dataset,depth=depth).to(rank)
    model_state_dict = model.state_dict()
    cnn_state_dict = remove_data_parallel(torch.load('./results_' + str(depth) + '_' + dataset + '/1_compatibility_fused.pth'))
    for key in cnn_state_dict:
        targetkey = 'target_'+key
        if targetkey in model_state_dict:
            model_state_dict[targetkey] = cnn_state_dict[key]
        else:
            model_state_dict[key] = cnn_state_dict[key]

    for key in model_state_dict:
        if ('target_'+key in model_state_dict) and ('area_conv' not in key) and ('conv_bias' not in key):
            model_state_dict[key] = model_state_dict['target_'+key]

    for name,param in model.named_parameters():
        if 'area_conv' in name:
            if any(str(x) in name[:-2] for x in areas):
                param.requires_grad = True
            else:
                param.requires_grad = False
        elif 'conv_bias' in name:
            if any(str(x) in name for x in areas):
                param.requires_grad = True 
            else:
                param.requires_grad = False
        else:
            param.requires_grad = False
        if 'target_' in name:
            param.requires_grad = False
    
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

    model.load_state_dict(model_state_dict,strict=False)
    model = DDP(model, device_ids=[rank], output_device=rank)
    torch.save(model.state_dict(),'./results_' + str(depth) + '_' + dataset + '/2_inverse_0.pth')

    optimizer = optim.AdamW(model.parameters(),lr=0.0001)
    optimizer.zero_grad(set_to_none=True)
    
    log_size = 1000
    train_size, test_size = get_size(dataset)

    for epoch in range(100):
        dataloader.sampler.set_epoch(epoch)  

        if rank == 0:
            start = time.time()
            fullstart = time.time()
            batch_errors = np.zeros(depth)
            batch_eigens = np.zeros(depth)
            total_errors = np.zeros(depth)
            total_eigens = np.zeros(depth)

        for i, data in enumerate(dataloader):
            if dataset in ['fashionmnist','cifar10','cifar100']:
                inputs, labels = cutmix_or_mixup(data)
            else:
                inputs, labels = data
            inputs, labels = inputs.to(rank), labels.to(rank)
            errors, eigens = model(inputs,rank,areas)
            full_loss = errors.mean() + 1e1*eigens.mean()
            full_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
            optimizer.step()
            optimizer.zero_grad()

            # ----------------------------------------------------------
            torch.cuda.set_device(rank)
            errors_ = [None for _ in range(world_size)]
            eigens_ = [None for _ in range(world_size)]
            dist.all_gather_object(errors_, errors)
            dist.all_gather_object(eigens_, eigens)
            
            if rank == 0:
                total_errors += 0.5*(errors_[0].cpu().detach().numpy() + errors_[1].cpu().detach().numpy())/train_size
                total_eigens += 0.5*(eigens_[0].cpu().detach().numpy() + eigens_[1].cpu().detach().numpy())/train_size
                batch_errors += 0.5*(errors_[0].cpu().detach().numpy() + errors_[1].cpu().detach().numpy())/log_size
                batch_eigens += 0.5*(eigens_[0].cpu().detach().numpy() + eigens_[1].cpu().detach().numpy())/log_size

                if i%(log_size) == (log_size-1):
                    print('Epoch',str(epoch+1),'Training batch',str(int((i+1))),f'Time taken: {time.time() - start:.5f}', flush=True)
                    print('Loss:',printarray(batch_errors), flush=True)
                    print('Regs:',printarray(batch_eigens), flush=True)
                    start = time.time()
                    batch_errors = np.zeros(depth)
                    batch_eigens = np.zeros(depth)

        if rank == 0:
            print('Epoch',str(epoch+1),f'Time taken: {time.time() - fullstart:.5f}', flush=True)
            print('Loss:',printarray(total_errors), flush=True)
            print('Regs:',printarray(total_eigens), flush=True)
            start = time.time()
            total_errors = np.zeros(8)
            total_eigens = np.zeros(8)
            torch.save(model.state_dict(),'./results_' + str(depth) + '_' + dataset + '/2_inverse_' + str(epoch+1) + '.pth')
    cleanup()

if __name__ == '__main__':
    world_size = 2   
    mp.spawn(main,args=(world_size,),nprocs=world_size)