import torch
import torchvision.transforms as transforms

def get_data(data_set, download=False, workers=8, root='../data', batch_size=64):
    trans = transforms.ToTensor()
    train_set = data_set(root=root, train=True, transform=trans, download=download)
    test_set = data_set(root=root, train=False, transform=trans)
    
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True, pin_memory=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False, pin_memory=True, num_workers=8)

    return train_loader, test_loader
