# Deep Learning Final Project

- Team name: Count on Me
- Contributors
    - 백제훈
    - 이윤주
    - 조승혁
    - 허성우
- Topic: [Sentiment Classification](https://www.kaggle.com/c/sentence-classification/overview)

## Instructions
### Setup
You can setup `pytorch-sentiment-classification` by executing the following commands.
```
git clone https://github.com/seunghyukcho/pytorch-sentiment-classification.git
cd pytorch-sentiment-classification
pip install -r requirements.txt
```

You can ask me for other setups for example, python and CUDA.

### New Tokenizer
1. Run `cp tokenizers/char.py tokenizers/<your_tokenizer_name>.py`.
2. In `tokenizers/<your_tokenizer_name>.py`, modify `add_tokenizer_args` and `Tokenizer`. You can add more functions but ***never modify the existing functions' signature!***
3. Add `<your_tokenizer_name>` to `arguments.py` `tokenizers` array.

### New Model
1. Run `cp models/rnn.py models/<your_model_name>.py`.
2. In `models/<your_model_name>.py`, modify `add_model_args` and `Model`. You can add more functions but ***never modify the existing functions' signature!***
3. Add `<your_model_name>` to `arguments.py` `models` array.

### Train
You can use `python train.py --model <model> --tokenizer <tokenizer> -h` to look at the arguments.
Notice that you must pass `--model`, `--tokenizer` arguments to see the help message.

```
$ python train.py --model rnn --tokenizer char --help

usage: train.py [-h] [--seed SEED] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR] [--log_interval LOG_INTERVAL] [--train_data TRAIN_DATA] [--valid_data VALID_DATA] [--n_classes N_CLASSES] [--device {cpu,cuda}] [--save_dir SAVE_DIR] [--log_dir LOG_DIR] [--model {rnn}] [--tokenizer {char}] [--n_hids N_HIDS]
                [--n_layers N_LAYERS] [--embd_dim EMBD_DIM]

optional arguments:
  -h, --help            show this help message and exit
  --model {rnn}
  --tokenizer {char}

train:
  --seed SEED           Random seed.
  --epochs EPOCHS       Number of epochs for training.
  --batch_size BATCH_SIZE
                        Number of instances in a batch.
  --lr LR               Learning rate.
  --log_interval LOG_INTERVAL
                        Log interval.
  --train_data TRAIN_DATA
                        Root directory of train data.
  --valid_data VALID_DATA
                        Root directory of validation data.
  --n_classes N_CLASSES
                        Number of classes.
  --device {cpu,cuda}   Device going to use for training.
  --save_dir SAVE_DIR   Folder going to save model checkpoints.
  --log_dir LOG_DIR     Folder going to save logs.

model:
  --n_hids N_HIDS       Number of hidden units
  --n_layers N_LAYERS   Number of layers
  --embd_dim EMBD_DIM   Embedding dimension
```

### Inference 
You can make predictions on a sample input data.
Running `inference.py` will create the predictions at `submit.csv`.

```
$ python inference.py --help

usage: inference.py [-h] [--batch_size BATCH_SIZE] [--test_data TEST_DATA] [--device {cpu,cuda}] [--ckpt_dir CKPT_DIR]

optional arguments:
  -h, --help            show this help message and exit

test:
  --batch_size BATCH_SIZE
                        Number of instances in a batch.
  --test_data TEST_DATA
                        Root directory of test data.
  --device {cpu,cuda}   Device going to use for training.
  --ckpt_dir CKPT_DIR   Directory which contains the checkpoint and args.json.
```

