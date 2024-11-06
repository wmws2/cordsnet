import torch
import torchvision
from torchvision import datasets,transforms
from torchvision.transforms import v2
from collections import OrderedDict

channels = [64,64,128,128,256,256,512,512]
sizes = [56,56,28,28,14,14,7,7]

# when training across multiple nodes and multiple GPUs, a prefix is added to all layer names
# call this function on the model state dict to remove prefixes so that you can load this on a single GPU for testing
def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    
    return new_state_dict

# loading datasets with augmentations
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