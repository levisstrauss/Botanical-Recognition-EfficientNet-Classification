import argparse
import torch
import json
import os
import torchvision
from torch import nn
from PIL import Image
import json
from model_utils import setup_model
from utils import process_image
import numpy as np

def get_predict_args():
    """
    Retrieves and parses the command line arguments for prediction
    Returns:
        parse_args() - data structure that stores the command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Predict flower name from an image'
    )
    
    # Required arguments
    parser.add_argument('input', type=str, 
                        help='Path to image file')
    
    parser.add_argument('checkpoint', type=str, 
                        help='Path to checkpoint file')
    
    parser.add_argument('--top_k', type=int, default=5,
                        help='Return top K most likely classes')
    
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='Path to JSON file mapping categories to real names')
    
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for inference if available')
    
    return parser.parse_args()


def load_checkpoint(filepath: str, device: torch.device)-> torch.nn.Module:
    """
    Load a saved checkpoint and rebuild the model
    
    Args:
        filepath: Path to checkpoint file
        device: Device to load model to
    
    Returns:
        model: Loaded model
    """
    
    print(f"Loading checkpoint from {filepath}...")
    checkpoint = torch.load(filepath, map_location=device)
    
    # Initialize model with appropriate architecture
    model = setup_model(arch=checkpoint['architecture'], device=device)
    
    # Load the state dict in the modal
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    print("Checkpoint loaded successfully!")
    print(f"Architecture: {checkpoint['architecture']}")
    print(f"Number of classes: {len(model.class_to_idx)}") 
    print(f"Epochs completed: {checkpoint['epochs_completed']}")
    
    return model


def predict(image_path: str, model: torch.nn.Module, device: torch.device, top_k: int = 5):
    """
    Predict the class of an image using a trained deep learning model.
    
    Args:
        image_path: Path to image file
        model: Trained model
        device: Device to run prediction on
        top_k: Number of top predictions to return
        
    Returns:
        top_probs: List of top K probabilities
        top_classes: List of top K class indices
    """
    # Move model to evaluation mode
    model.eval()

    model.to(device)
    
    # Process image
    img = process_image(image_path)
    img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(device)
    
    # Calculate predictions
    with torch.inference_mode():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        # Get top K probabilities and indices
        top_probs, top_indices = probabilities.topk(top_k)
        
        # Convert to lists
        top_probs = top_probs.squeeze().cpu().numpy().tolist()
        top_indices = top_indices.squeeze().cpu().numpy().tolist()
        
        # Convert indices to classes
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_classes = [idx_to_class[idx] for idx in top_indices]
        
    return top_probs, top_classes

def main():
    # Get command line arguments
    args = get_predict_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    print(f"Using device: {device}")
    
    # Load categories to names mapping if provided
    if args.category_names and os.path.exists(args.category_names):
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
    else:
        cat_to_name = None
        print(f"Warning: Category names file not found at {args.category_names}")
    
    # Load model from checkpoint
    model = load_checkpoint(args.checkpoint, device)
    
    # Make prediction
    print(f"\nMaking prediction for image: {args.input}")
    probs, classes = predict(args.input, model, device, args.top_k)
    
    # Print results
    print(f"\nTop {args.top_k} predictions:")
    for i, (prob, class_idx) in enumerate(zip(probs, classes)):
        if cat_to_name:
            print(f"{cat_to_name[class_idx]}: {prob*100:.2f}%")
        else:
            print(f"Class {class_idx}: {prob*100:.2f}%")

if __name__ == '__main__':
    main()