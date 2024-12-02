import argparse
import torch
from torch import nn, optim
from model_utils import setup_model, save_checkpoint
from data_utils import load_data
from train_utils import engine


def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using the argparse module.
    """
    parser = argparse.ArgumentParser(description='Train a neural network on a dataset')

    # Required argument - data directory
    parser.add_argument('data_directory',
                        default="flowers",
                        help='Path to data directory containing train, valid, and test folders')

    # Name of the folder where will be store the model
    parser.add_argument('--save_dir',
                        type=str,
                        default='checkpoints',
                        help='Directory to save checkpoints')

    parser.add_argument('--arch',
                        type=str,
                        default='efficientnet_b0',
                        choices=['efficientnet_b0'],
                        help='Model architecture')

    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.001,
                        help='Learning rate')

    parser.add_argument('--epochs',
                        type=int,
                        default=5,
                        help='Number of epochs')

    parser.add_argument('--gpu',
                        action='store_true',
                        help='Use GPU for training if available')

    return parser.parse_args()


def main():
    # Get arguments
    args = get_input_args()

    # Set device
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the data directory
    data_dir = args.data_directory
    print(f"Data Directory: {data_dir}")

    # Model architecture
    arch = args.arch
    print(f"Model Architecture: {arch}")

    # Learning rate
    learning_rate = args.learning_rate
    print(f"Learning Rate: {learning_rate}")

    # Number of epochs
    epochs = args.epochs
    print(f"Number of Epochs: {epochs}")

    # Save directory
    save_dir = args.save_dir
    print(f"Checkpoints will be saved in: {save_dir}")

    # Load data
    print("Loading data...")
    dataloaders, image_datasets = load_data(args.data_directory)

    # Setup model
    print(f"Setting up {args.arch} model...")
    model = setup_model(args.arch, device=device)

    #------------- Define criterion and optimizer-----------#
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    #---------------- Train model ------------------------#
    print("Starting training...")
    results = engine(
        model=model,
        train_dataloader=dataloaders['train'],
        test_dataloader=dataloaders['valid'],
        optimizer=optimizer,
        loss_fn=criterion,
        epochs=args.epochs,
        device=device
    )

    #------------------- Save checkpoint ------------------#
    checkpoint_path = save_checkpoint(model, image_datasets, optimizer, args, args.save_dir)
    print(f"\nCheckpoint saved to: {checkpoint_path}")


if __name__ == '__main__':
    main()
