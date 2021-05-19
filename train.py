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

from dataset import Dataset, PadBatch
from arguments import get_task_parser, add_train_args

# stratified split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold

# confusion matrix
from sklearn.metrics import confusion_matrix

# convolution function of confusion matrix --> precisions, accuracy
def convolution(cm):
  eps = 1e-6
  rec = np.diag(cm)/(cm.sum(axis=1)+eps)
  prec = np.diag(cm)/(cm.sum(axis=0)+eps)
#   prec = np.diag(cm/cm.sum(axis=1))
  acc = np.diag(cm).sum()/np.sum(cm)
  return prec, rec ,acc

# cost sensitive loss function
# adjust misclassification loss 
# the further prediction and ground truth get, the larger the loss is.
# cost matrix 'M' is normalized by division by its the max value 4.
def cost_sensitive_loss(input, target, M):
    device = input.device
    M = M.to(device)
    return (M[target, :]*input.float()).sum(axis=-1)

class CostSensitiveLoss(nn.Module):
    def __init__(self, exp=1, reduction='mean'):
        super(CostSensitiveLoss, self).__init__()
        self.normalization = nn.Softmax(dim=1)
        self.reduction = reduction
        # x = np.abs(np.arange(5, dtype=np.float32))
        # M = np.abs((x[:, np.newaxis] - x[np.newaxis, :])) ** exp
        # M /= M.max()

        M = np.array(
          [[0,1,1,1,1.1],
            [1,0,1,1,1.1],
            [1.1,1,0,1,1.1],
            [1.1,1,1,0,1],
            [1.1,1,1,1,0]]
        )
        M = M/M.max()
        self.M = torch.from_numpy(M)

    def forward(self, probs, target):
        preds = self.normalization(probs)
        loss = cost_sensitive_loss(preds, target, self.M)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError('\'reduction\' should be among mean, sum, none.')

class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y),1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0,int(1e8),size=()).item()
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return len(self.y)
                    
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
    getattr(model_module, 'add_model_args')(parser)
    getattr(tokenizer_module, 'add_tokenizer_args')(parser)
    args = parser.parse_args()

    # Seed settings
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print('Loading tokenizer...')
    tokenizer = tokenizer(args)

    print('Loading train dataset...')
    train_dataset = Dataset(args.train_data, tokenizer=tokenizer, label=True)
    y = torch.from_numpy(train_dataset.data.loc[:,'Category'].values)
    sampler = StratifiedBatchSampler(y, args.batch_size)
    # train_loader = DataLoader(dataset=train_dataset, collate_fn=PadBatch(), batch_sampler= sampler)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=PadBatch())

    print('Loading validation dataset...')
    valid_dataset = Dataset(args.valid_data, tokenizer=tokenizer, label=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, collate_fn=PadBatch())
    
    print('Building model...')
    model = model(args, vocab_size=tokenizer.get_vocab_size() + 1)
    model = model.to(args.device)

    # Ignore annotators labeling which is -1
#     criterion = CostSensitiveLoss(exp=1, reduction='mean')
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = Adam(model.parameters(), lr=args.lr)

    print('Start training!')
    best_accuracy = 0
    writer = SummaryWriter(args.log_dir)
    for epoch in range(args.epochs):
        train_loss = 0.0
        train_correct = 0
        train_cm = np.zeros(shape=(5,5)) # confusion matrix
        model.train()
        for x, y, lens in train_loader:
            model.zero_grad()

            # Move the parameters to device given by argument
            x, y, lens = x.to(args.device), y.to(args.device), lens.to(args.device)
            pred = model(x, lens)

            # Calculate loss of annotators' labeling
            loss = criterion(pred, y.view(-1))

            # Update model weight using gradient descent
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Calculate classifier accuracy
            pred = torch.argmax(pred, dim=1)
            train_correct += torch.sum(torch.eq(pred, y)).item()
            train_cm += confusion_matrix( y.cpu().numpy(),pred.cpu().numpy(),labels=[0,1,2,3,4])
            

        # Validation
        with torch.no_grad():
            valid_correct = 0
            valid_cm = np.zeros(shape= (5,5)) # confusion matrix for validation set
            model.eval()
            for x, y, lens in valid_loader:
                x, y, lens = x.to(args.device), y.to(args.device), lens.to(args.device)
                pred = model(x, lens)
                pred = torch.argmax(pred, dim=1)
                valid_correct += torch.sum(torch.eq(pred, y)).item()
                valid_cm += confusion_matrix( y.cpu().numpy(), pred.cpu().numpy(), labels=[0,1,2,3,4])
        
        prec_tr, rec_tr, acc_tr = convolution(train_cm)
        prec_val, rec_val, acc_val = convolution(valid_cm)
        
        print(
            f'Epoch: {(epoch + 1):4d} | '
            f'Train Loss: {train_loss:.3f} | '
            f'Train Accuracy: {(train_correct / len(train_dataset)):.2f} | '
            f'Valid Accuracy: {(valid_correct / len(valid_dataset)):.2f}'
        )
        print()
        print('Train confusion matrix')
        print(train_cm)
        print()
        print('validation confusion matrix')
        print(valid_cm)
        print('-------------------\n')
        # Save tensorboard log
        if epoch % args.log_interval == 0:
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_accuracy', acc_tr, epoch)
            writer.add_scalar('valid_accuracy', acc_val, epoch)

        # Save tensorboard log
        if epoch % args.log_interval == 0:
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_accuracy', acc_tr, epoch)
            writer.add_scalar('valid_accuracy', acc_val, epoch)

        # Save the model with highest accuracy on validation set
        if best_accuracy < acc_val:
            best_accuracy = acc_val
            checkpoint_dir = Path(args.save_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model': model.state_dict()
            }, checkpoint_dir / 'best_model.pth')

            with open(checkpoint_dir / 'args.json', 'w') as f:
                json.dump(args.__dict__, f, indent=2)

