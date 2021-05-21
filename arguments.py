import argparse

models = ['rnn','lstm','drnn']
tokenizers = ['char', 'word','wordk']


def get_task_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model', type=str, choices=models)
    parser.add_argument('--tokenizer', type=str, choices=tokenizers)
    return parser


def add_train_args(parser):
    group = parser.add_argument_group('train')
    group.add_argument('--seed', type=int, default=7777,
                       help="Random seed.")
    group.add_argument('--epochs', type=int, default=10,
                       help="Number of epochs for training.")
    group.add_argument('--batch_size', type=int, default=32,
                       help="Number of instances in a batch.")
    group.add_argument('--lr', type=float, default=1e-5,
                       help="Learning rate.")
    group.add_argument('--log_interval', type=int, default=10,
                       help="Log interval.")
    group.add_argument('--train_data', type=str,
                       help="Root directory of train data.")
    group.add_argument('--valid_data', type=str,
                       help="Root directory of validation data.")
    group.add_argument('--n_classes', type=int, default=5,
                       help="Number of classes.")
    group.add_argument('--device', type=str, choices=['cpu', 'cuda'],
                       help="Device going to use for training.")
    group.add_argument('--save_dir', type=str, default='checkpoints/',
                       help="Folder going to save model checkpoints.")
    group.add_argument('--log_dir', type=str, default='logs/',
                       help="Folder going to save logs.")
    parser.add_argument('--model', type=str, choices=models)
    parser.add_argument('--tokenizer', type=str, choices=tokenizers)


def add_test_args(parser):
    group = parser.add_argument_group('test')
    group.add_argument('--batch_size', type=int, default=32,
                       help="Number of instances in a batch.")
    group.add_argument('--test_data', type=str,
                       help="Root directory of test data.")
    group.add_argument('--device', type=str, choices=['cpu', 'cuda'],
                       help="Device going to use for training.")
    group.add_argument('--ckpt_dir', type=str,
                       help="Directory which contains the checkpoint and args.json.")

