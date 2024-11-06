import torch
import torchvision
from torchvision import datasets,transforms
from torchvision.transforms import v2
from collections import OrderedDict

channels = [64,64,128,128,256,256,512,512]
sizes = [56,56,28,28,14,14,7,7]

def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    
    return new_state_dict

def load_dataset(name,split):

    if name == 'imagenet':
        if split == 'train':
            dataset = datasets.ImageFolder('./imagenet/train',transforms.Compose([
                v2.RandomResizedCrop(224),
                v2.RandomHorizontalFlip(),
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                v2.RandAugment(),
                v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ]))
        elif split == 'test':
            dataset = datasets.ImageFolder('./imagenet/val',transforms.Compose([
                v2.Resize(256),
                v2.CenterCrop(224),
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ]))
    
    if name == 'cifar10':
        if split == 'train':
            dataset = torchvision.datasets.CIFAR10(root='./cifar10', download=True, train=True, transform=transforms.Compose([
                v2.Resize(224),
                v2.RandomHorizontalFlip(),
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                v2.RandAugment(),
                v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
            ]))
        elif split == 'test':
            dataset = torchvision.datasets.CIFAR10(root='./cifar10', download=True, train=False, transform=transforms.Compose([
                v2.Resize(224),
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
            ]))

    if name == 'cifar100':
        if split == 'train':
            dataset = torchvision.datasets.CIFAR100(root='./cifar100', download=True, train=True, transform=transforms.Compose([
                v2.Resize(224),
                v2.RandomHorizontalFlip(),
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                v2.RandAugment(),
                v2.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
            ]))
        elif split == 'test':
            dataset = torchvision.datasets.CIFAR100(root='./cifar100', download=True, train=False, transform=transforms.Compose([
                v2.Resize(224),
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
            ]))

    if name == 'mnist':
        if split == 'train':
            dataset = torchvision.datasets.MNIST(root='./mnist', download=True, train=True, transform=transforms.Compose([
                v2.Resize(224),
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
                v2.ColorJitter(brightness=0.1, contrast=0.1),
                v2.ElasticTransform(alpha=20, sigma=5),
                v2.Normalize(mean=[0.1307], std=[0.3081])
            ]))
        elif split == 'test':
            dataset = torchvision.datasets.MNIST(root='./mnist', download=True, train=False, transform=transforms.Compose([
                v2.Resize(224),
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.1307], std=[0.3081])
            ]))

    if name == 'fashionmnist':
        if split == 'train':
            dataset = torchvision.datasets.FashionMNIST(root='./fashionmnist', download=True, train=True, transform=transforms.Compose([
                v2.Resize(224),
                v2.RandomHorizontalFlip(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
                v2.ColorJitter(brightness=0.1, contrast=0.1),
                v2.Normalize(mean=[0.2860], std=[0.3530])
            ]))
        elif split == 'test':
            dataset = torchvision.datasets.FashionMNIST(root='./fashionmnist', download=True, train=False, transform=transforms.Compose([
                v2.Resize(224),
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.2860], std=[0.3530])
            ]))

    return dataset

def load_pretrained(dataset):

    pretrained = torch.load('./results_8_' + dataset + '/0_resnet18.pth')

    todel = []
    if not dataset == 'imagenet':
        for key in pretrained:
            if 'out_fc' in key:
                todel.append(key)

    if dataset in ['mnist','fashionmnist']:
        for key in pretrained:
            if 'inp_conv' in key:
                todel.append(key)

    for key in todel:
        del pretrained[key]

    return pretrained

def get_size(dataset):

    if dataset in ['mnist','fashionmnist']:
        train, test = 60000, 10000
    if dataset in ['cifar10','cifar100'] :
        train, test = 50000, 10000
    if dataset in ['imagenet']:
        train, test = 1281167, 50000
    
    return train,test


def load_dataset_dense(name,split):

    if name == 'imagenet':
        if split == 'train':
            dataset = datasets.ImageFolder('./imagenet/train',transforms.Compose([
                v2.RandomResizedCrop(224),
                v2.RandomHorizontalFlip(),
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                v2.RandAugment(),
                v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ]))
        elif split == 'test':
            dataset = datasets.ImageFolder('./imagenet/val',transforms.Compose([
                v2.Resize(256),
                v2.CenterCrop(224),
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ]))
    
    if name == 'cifar10':
        if split == 'train':
            dataset = torchvision.datasets.CIFAR10(root='./cifar10', download=True, train=True, transform=transforms.Compose([
                v2.Resize(32),
                v2.RandomHorizontalFlip(),
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                v2.RandAugment(),
                v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
            ]))
        elif split == 'test':
            dataset = torchvision.datasets.CIFAR10(root='./cifar10', download=True, train=False, transform=transforms.Compose([
                v2.Resize(32),
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
            ]))

    if name == 'cifar100':
        if split == 'train':
            dataset = torchvision.datasets.CIFAR100(root='./cifar100', download=True, train=True, transform=transforms.Compose([
                v2.Resize(32),
                v2.RandomHorizontalFlip(),
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                v2.RandAugment(),
                v2.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
            ]))
        elif split == 'test':
            dataset = torchvision.datasets.CIFAR100(root='./cifar100', download=True, train=False, transform=transforms.Compose([
                v2.Resize(32),
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
            ]))

    if name == 'mnist':
        if split == 'train':
            dataset = torchvision.datasets.MNIST(root='./mnist', download=True, train=True, transform=transforms.Compose([
                v2.Resize(28),
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
                v2.ColorJitter(brightness=0.1, contrast=0.1),
                v2.ElasticTransform(alpha=20, sigma=5),
                v2.Normalize(mean=[0.1307], std=[0.3081])
            ]))
        elif split == 'test':
            dataset = torchvision.datasets.MNIST(root='./mnist', download=True, train=False, transform=transforms.Compose([
                v2.Resize(28),
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.1307], std=[0.3081])
            ]))

    if name == 'fashionmnist':
        if split == 'train':
            dataset = torchvision.datasets.FashionMNIST(root='./fashionmnist', download=True, train=True, transform=transforms.Compose([
                v2.Resize(28),
                v2.RandomHorizontalFlip(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
                v2.ColorJitter(brightness=0.1, contrast=0.1),
                v2.Normalize(mean=[0.2860], std=[0.3530])
            ]))
        elif split == 'test':
            dataset = torchvision.datasets.FashionMNIST(root='./fashionmnist', download=True, train=False, transform=transforms.Compose([
                v2.Resize(28),
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.2860], std=[0.3530])
            ]))

    return dataset