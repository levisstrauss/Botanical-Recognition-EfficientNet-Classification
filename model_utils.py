import torch
from torch import nn
import torchvision
from PIL import Image  
import os  

def setup_model(arch: str, device: torch.device):
    """
    Set up model architecture and modify classifier based on architecture choice
    """
    if arch == 'efficientnet_b0':
        
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        model = torchvision.models.efficientnet_b0(weights=weights)
        
        # Freeze feature parameters
        for param in model.features.parameters():
            param.requires_grad = False
        # Modify classifier
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, out_features=102, bias=True)
        )
    # elif arch == 'vgg13':
    #     weights = torchvision.models.VGG13_Weights.DEFAULT
    #     model = torchvision.models.vgg13(weights=weights)
    #     # Freeze feature parameters
    #     for param in model.features.parameters():
    #         param.requires_grad = False
    #     # Modify classifier
    #     model.classifier = nn.Sequential(
    #         nn.Linear(25088, hidden_units),
    #         nn.ReLU(),
    #         nn.Dropout(0.5),
    #         nn.Linear(hidden_units, 102)  # 102 flower classes
    #     )
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    
    return model.to(device)

def save_checkpoint(model, image_datasets, optimizer, args, save_dir):
    """Save model checkpoint"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    checkpoint = {
        'architecture': args.arch,
        'state_dict': model.state_dict(),
        'class_to_idx': image_datasets['train'].class_to_idx,
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs_completed': args.epochs,
    }
    
    checkpoint_path = os.path.join(save_dir, f'{args.arch}_checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path