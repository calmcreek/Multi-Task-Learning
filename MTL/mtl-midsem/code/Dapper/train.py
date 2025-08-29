# === FILE: code/Dapper/train.py ===
import yaml
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset import DapperDataset
from mtl_model import DapperMTL
from utils import set_seed, save_checkpoint
import os

def load_config(path='code/Dapper/config.yaml'):
    with open(path,'r') as f:
        return yaml.safe_load(f)

def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = {k: torch.stack([b[1].get(k, torch.tensor(0.0)) for b in batch]) for k in batch[0][1].keys()}
    masks = {k: torch.stack([b[2].get(k, torch.tensor(0.0)) for b in batch]) for k in batch[0][2].keys()}
    return xs, ys, masks

def train():
    cfg = load_config()
    set_seed(42)

    # Dataset
    ds = DapperDataset(cfg['paths']['processed_out'], task_cols=cfg['tasks'])
    dl = DataLoader(ds, batch_size=cfg['training']['batch_size'], shuffle=True, collate_fn=collate_fn)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Get input dimension
    sample = next(iter(dl))
    input_dim = sample[0].shape[1]

    # Model
    model = DapperMTL(input_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])
    mse = nn.MSELoss(reduction='none')

    for epoch in range(cfg['training']['epochs']):
        model.train()
        losses = []
        for xb, ys, masks in dl:
            xb = xb.to(device)
            ys = {k:v.to(device) for k,v in ys.items()}
            masks = {k:v.to(device) for k,v in masks.items()}

            preds = model(xb)

            # Masked MSE loss per task
            loss_val = mse(preds['valence'], ys['valence']) * masks['valence']
            loss_aru = mse(preds['arousal'], ys['arousal']) * masks['arousal']
            loss_pos = mse(preds['panas_pos'], ys['panas_pos']) * masks['panas_pos']

            loss = (loss_val.sum() / (masks['valence'].sum()+1e-8) +
                    loss_aru.sum() / (masks['arousal'].sum()+1e-8) +
                    loss_pos.sum() / (masks['panas_pos'].sum()+1e-8))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"Epoch {epoch+1}/{cfg['training']['epochs']} avg_loss={sum(losses)/len(losses):.4f}")

        # Save checkpoint every 10 epochs
        if (epoch+1) % 10 == 0:
            save_checkpoint(model, f"results/dapper_mtl_epoch{epoch+1}.pth")

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    train()
