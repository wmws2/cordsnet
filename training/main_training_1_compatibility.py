import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
from class_rnn_1_compatibility import *
from utils import *
import time

dataset = 'imagenet'
depth = 8
device = torch.device("cuda")
dataset_train = load_dataset(dataset,'train')
dataset_test = load_dataset(dataset,'test')

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

batch_size = 256
log_size = 500
num_workers= 32
train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)

model = step1cordsnet(dataset=dataset, depth=depth).to(device)
if depth == 8:
    model.load_state_dict(load_pretrained(dataset),strict=False)
model = nn.DataParallel(model, device_ids=[0,1])
criterion = nn.CrossEntropyLoss(reduction='sum')
torch.save(model.state_dict(), './results_' + str(depth) + '_' + dataset + '/1_compatibility_0.pth')

train_size, test_size = get_size(dataset)

for steps in range(3):

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1/(10**steps),momentum=0.9,weight_decay=0.0001)
    scheduler = LinearLR(optimizer,total_iters=100)

    for epoch in range(100):  # loop over the dataset multiple times

        running_loss = 0.0
        batch_loss = 0.0
        batch_acc = 0.0
        verystart = time.time()
        start = time.time()

        for i, data in enumerate(train_dataloader, 0):
            model.train()
            if dataset in ['fashionmnist','cifar10','cifar100']:
                inputs, labels = cutmix_or_mixup(data)
            else:
                inputs, labels = data
            current_batch_size = inputs.size()[0]
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            x0 = []
            for j in range(8):
                x0.append(torch.zeros(current_batch_size,channels[j],sizes[j],sizes[j]).to(device))
            a0 = torch.zeros(current_batch_size).to(device)
            outputs, x1 = model(x0, inputs)
            loss = criterion(outputs, labels)
            full_loss = loss/current_batch_size
            full_loss.backward()
            optimizer.step()
            scheduler.step()
            batch_loss += loss.item()/(log_size*batch_size)
            if dataset in ['fashionmnist','cifar10','cifar100']:
                batch_acc += (outputs.argmax(1) == labels.argmax(1)).sum().item()/(log_size*batch_size)
            else:
                batch_acc += (outputs.argmax(1) == labels).sum().item()/(log_size*batch_size)
            running_loss += loss.item()/train_size
            if i%log_size==(log_size-1):
                print('training batch',i+1,f'timetaken: {time.time() - start:.5f} loss: {batch_loss:.5f} accuracy: {batch_acc:.5f} lr: {optimizer.param_groups[-1]["lr"]:.5f}',flush=True)
                start = time.time()
                batch_loss = 0.0
                batch_acc = 0.0

        with torch.no_grad():
            correct = 0
            model.eval()
            for i, data in enumerate(test_dataloader, 0):
                inputs, labels = data
                current_batch_size = inputs.size()[0]
                inputs = inputs.to(device)
                labels = labels.to(device)
                x0 = []
                for j in range(8):
                    x0.append(torch.zeros(current_batch_size,channels[j],sizes[j],sizes[j]).to(device))
                a0 = torch.zeros(current_batch_size).to(device)
                outputs, x1 = model(x0, inputs)
                correct += (outputs.argmax(1) == labels).sum().item()/test_size

            print(f'{epoch + 1} loss: {running_loss:.5f} testacc: {correct:.5f} timetaken: {time.time() - verystart:.5f}',flush=True)
            torch.save(model.state_dict(),'./results_' + str(depth) + '_' + dataset + '/1_compatibility_' + str(steps+1) + '_' + str(epoch+1) + '.pth')