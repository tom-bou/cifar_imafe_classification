import torch
import torchvision
import torchvision.transforms as transforms

# Set multiprocessing start method
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

def create_dataloader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                                  shuffle=True, num_workers=0)  # Set num_workers to 0 for debugging

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                                 shuffle=False, num_workers=0)  # Set num_workers to 0 for debugging

        return trainloader, testloader
    except Exception as e:
        print("Error in creating dataloaders:", e)
        raise