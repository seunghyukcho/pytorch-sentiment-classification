import json

import torch
import argparse
import importlib
import numpy as np
import torch.nn as nn
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import CoNAL
from dataset import Dataset
from arguments import get_task_parser, add_train_args


if __name__ == "__main__":
    # Read task argument first, and determine the other arguments
    task_parser = get_task_parser()

    task_parser = task_parser.parse_known_args()[0]
    model_name = task_parser.model
    model_module = importlib.import_module(f'models.{model_name}')
    model = getattr(model_module, 'Model')

    tokenizer_name = task_parser.tokenizer
    tokenizer_module = importlib.import_module(f'tokenizers.{tokenizer_name}')
    tokenizer = getattr(tokenizer_module, 'Tokenizer')

    parser = argparse.ArgumentParser()
    add_train_args(parser)
    getattr(model_module, 'add_model_args')
    getattr(tokenizer_module, 'add_tokenizer_args')
    args = parser.parse_args()

    # Seed settings
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print('Loading tokenizer...')
    tokenizer = tokenizer(args)

    print('Loading train dataset...')
    train_dataset = Dataset(args.train_data, tokenizer=tokenizer, label=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    print('Loading validation dataset...')
    valid_dataset = Dataset(args.valid_data, tokenizer=tokenizer, label=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size)

    print('Building model...')
    model = model(args, embd_dim=tokenizer.get_embedding_dimension())
    model = model.to(args.device)

    # Ignore annotators labeling which is -1
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = Adam(model.parameters(), lr=args.lr)

    print('Start training!')
    best_accuracy = 0
    writer = SummaryWriter(args.log_dir)
    for epoch in range(args.epochs):
        train_loss = 0.0
        train_correct = 0
        model.train()
        for x, y in train_loader:
            model.zero_grad()

            # Move the parameters to device given by argument
            x, y = x.to(args.device), y.to(args.device)
            pred = model(x)

            # Calculate loss of annotators' labeling
            loss = criterion(pred, y.view(-1))

            # Update model weight using gradient descent
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Calculate classifier accuracy
            pred = torch.argmax(pred, dim=1)
            train_correct += torch.sum(torch.eq(pred, y)).item()

        # Validation
        with torch.no_grad():
            valid_correct = 0
            model.eval()
            for x, y in valid_loader:
                x, y = x.to(args.device), y.to(args.device)
                pred = model(x)
                pred = torch.argmax(pred, dim=1)
                valid_correct += torch.sum(torch.eq(pred, y)).item()

        print(
            f'Epoch: {(epoch + 1):4d} | '
            f'Train Loss: {train_loss:.3f} | '
            f'Train Accuracy: {(train_correct / len(train_dataset)):.2f} | '
            f'Valid Accuracy: {(valid_correct / len(valid_dataset)):.2f}'
        )

        # Save tensorboard log
        if epoch % args.log_interval == 0:
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_accuracy', train_correct / len(train_dataset), epoch)
            writer.add_scalar('valid_accuracy', valid_correct / len(valid_dataset), epoch)

        # Save the model with highest accuracy on validation set
        if best_accuracy < valid_correct:
            best_accuracy = valid_correct
            checkpoint_dir = Path(args.save_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'auxiliary_network': model.auxiliary_network.state_dict(),
                'noise_adaptation_layer': model.noise_adaptation_layer.state_dict(),
                'classifier': model.classifier.state_dict()
            }, checkpoint_dir / 'best_model.pth')

            with open(checkpoint_dir / 'args.json', 'w') as f:
                json.dump(args.__dict__, f, indent=2)

