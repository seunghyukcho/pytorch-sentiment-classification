import json
import torch
import argparse
import importlib
from tqdm import tqdm
from pathlib import Path
from munch import munchify
from torch.utils.data import DataLoader

from arguments import add_test_args
from dataset import Dataset, PadBatch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_test_args(parser)
    args = parser.parse_args()

    print('Loading configurations...')
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_file = ckpt_dir / 'best_model.pth'
    ckpt = torch.load(ckpt_file)

    config_file = ckpt_dir / 'args.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
        config = munchify(config)

    model_name = config.model
    model_module = importlib.import_module(f'models.{model_name}')
    tokenizer_name = config.tokenizer
    tokenizer_module = importlib.import_module(f'tokenizers.{tokenizer_name}')
    tokenizer = getattr(tokenizer_module, 'Tokenizer')(config)

    test_dataset = Dataset(args.test_data, tokenizer=tokenizer)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, collate_fn=PadBatch(inference=True))

    model = getattr(model_module, 'Model')(config, vocab_size=tokenizer.get_vocab_size() + 1, sentence_size= tokenizer.get_sentence_size() + 1)
    model.load_state_dict(ckpt['model'])
    model = model.to(args.device)

    with torch.no_grad():
        preds = torch.empty(0).to(args.device)
        model.eval()
        for w, s, lw, ls in tqdm(test_loader):
            w, s, lw, ls = w.to(args.device), s.to(args.device), lw.to(args.device), ls.to(args.device)
            
            # todo
            pred = model(x, lens)
            pred = torch.argmax(pred, dim=1)
            preds = torch.cat([preds, pred], dim=0)
        
        with open('submit.csv', 'w') as f:
            f.write('Id,Category\n')
            for i in range(preds.size(0)):
                f.write(str(i) + ',' + str(int(preds[i].item())) + '\n')

