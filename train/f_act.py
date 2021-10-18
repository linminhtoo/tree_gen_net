import argparse
import logging
import os
import pickle
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_dir)
from data_scripts.utils import seed_everything
from dataset.dataset import FingerprintDataset
from model.basic import BasicFeedforward


def train(args):
    seed_everything(args.random_seed)

    # load templates
    with open(args.path_templates, 'r') as f:
        template_strs = [l.strip().split('|')[1] for l in f.readlines()]

    # instantiate model & loss function
    if args.model == "f_act":
        model = BasicFeedforward(
            input_size=args.input_dim * 3,
            act_fn="ReLU",
            hidden_sizes=args.hidden_sizes, # [1000, 1200, 3000, 3000]
            output_size=4,
            dropout=args.dropout, # 0
            final_act_fn="softmax"
        )
        criterion = nn.CrossEntropyLoss(reduction='sum')

    elif args.model == "f_rxn":
        model = BasicFeedforward(
            input_size=args.input_dim * 4,
            act_fn="ReLU",
            hidden_sizes=args.hidden_sizes, # [1000, 1200, 3000, 3000]
            output_size=len(template_strs),
            dropout=args.dropout, # 0
            final_act_fn="softmax"
        )
        criterion = nn.CrossEntropyLoss(reduction='sum')

    elif args.model == "f_rt1":
        model = BasicFeedforward(
            input_size=args.input_dim * 3,
            act_fn="ReLU",
            hidden_sizes=args.hidden_sizes, # [1000, 1200, 3000, 3000]
            output_size=256,
            dropout=args.dropout, # 0
            final_act_fn=None
        )
        criterion = nn.MSELoss(reduction='sum')

    elif args.model == "f_rt2":
        model = BasicFeedforward(
            input_size=args.input_dim * 4 + len(template_strs),
            act_fn="ReLU",
            hidden_sizes=args.hidden_sizes, # [1000, 1200, 3000, 3000]
            output_size=256,
            dropout=args.dropout, # 0
            final_act_fn=None
        )
        criterion = nn.MSELoss(reduction='sum')

    else:
        raise ValueError(f"invalid model name {args.model}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using {device} device')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate) # 1e-4

    # prepare train & val datasets
    train_dataset = FingerprintDataset(args.path_steps_train, args.path_states_train)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True) # bs = 64

    valid_dataset = FingerprintDataset(args.path_steps_val, args.path_states_val)
    valid_loader = DataLoader(valid_dataset, batch_size=args.bs_eval, shuffle=False) # 64
    del train_dataset, valid_dataset

    # start training
    train_losses, valid_losses = [], []
    train_accs, valid_accs = [], []
    max_valid_acc = float('-inf')
    wait = 0 # early stopping patience counter

    start = time.time()
    for epoch in range(args.epochs):
        train_loss, train_correct, train_seen = 0, 0, 0
        train_loader = tqdm(train_loader, desc='training')
        model.train()

        for data in train_loader:
            X, y = data
            X, y = X.to(device), y.to(device)

            if args.model == "f_rt1" or args.model == "f_rt2":
                # split y into mol_embedding + mol_idx
                rct_idxs = y[:, -1] # lost a dim
                y = y[:, :-1]

            model.zero_grad()
            optimizer.zero_grad()

            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_seen += y.shape[0]

            if args.model == "f_act" or args.model == "f_rxn":
                # calculate accuracy
                _, batch_preds = torch.topk(y_pred, k=1, dim=1)
                train_correct += torch.sum(torch.eq(batch_preds, y), dim=0).item()

            else:
                # nearest neighbor search over all building block embeddings to get nearest idx
                # compare with groundtruth building block idx
                pred_idxs = nn_search(y_pred, rct_embeddings, k=1)
                train_correct += torch.sum(torch.eq(pred_idxs, rct_idxs), dim=0).item()

            train_loader.set_description(f"training: loss={train_loss/train_seen:.4f}, top-1 acc={train_correct/train_seen:.4f}")
            train_loader.refresh()

        train_losses.append(train_loss/train_seen)
        train_accs.append(train_correct/train_seen)

        # validation
        model.eval()
        with torch.no_grad():
            valid_loss, valid_correct, valid_seen = 0, 0, 0
            valid_loader = tqdm(valid_loader, desc='validating')

            for data in valid_loader:
                X, y = data
                X, y = X.to(device), y.to(device)

                if args.model == "f_rt1" or args.model == "f_rt2":
                    # split y into mol_embedding + mol_idx
                    rct_idxs = y[:, -1] # lost a dim
                    y = y[:, :-1]

                y_pred = model(X)
                loss = criterion(y_pred, y)

                valid_loss += loss.item()
                valid_seen += y.shape[0]

                if args.model == "f_act" or args.model == "f_rxn":
                    # calculate accuracy
                    _, batch_preds = torch.topk(y_pred, k=1, dim=1)
                    valid_correct += torch.sum(torch.eq(batch_preds, y), dim=0).item()

                else:
                    # nearest neighbor search over all building block embeddings to get nearest idx
                    # compare with groundtruth building block idx
                    pred_idxs = nn_search(y_pred, rct_embeddings, k=1)
                    valid_correct += torch.sum(torch.eq(pred_idxs, rct_idxs), dim=0).item()

                valid_loader.set_description(f"validating: loss={valid_loss/valid_seen:.4f}, top-1 acc={valid_correct/valid_seen:.4f}")
                valid_loader.refresh()

        valid_losses.append(valid_loss/valid_seen)
        valid_accs.append(valid_correct/valid_seen)

        if args.checkpoint and valid_accs[-1] > max_valid_acc:
            # checkpoint model
            model_state_dict = model.state_dict()
            checkpoint_dict = {
                "epoch": epoch,
                "state_dict": model_state_dict, "optimizer": optimizer.state_dict(),
                "train_accs": train_accs, "train_losses": train_losses,
                "valid_accs": valid_accs, "valid_losses": valid_losses,
                "max_valid_acc": max_valid_acc
            }
            checkpoint_filename = (
                args.checkpoint_folder
                / f"{args.expt_name}.pth.tar"
            )
            torch.save(checkpoint_dict, checkpoint_filename)

        if args.early_stop and max_valid_acc - valid_accs[-1] > 0:
            if args.early_stop_patience <= wait:
                message = f"\nEarly stopped at the end of epoch: {epoch}, \
                \nvalid loss: {valid_losses[-1]:.4f}, valid top-1 acc: {valid_accs[-1]:.4f}\n"
                logging.info(message)
                break
            else:
                wait += 1
                logging.info(
                    f'\nValidation accuracy did not increase, patience count: {wait} \
                    \n'
                )
        else:
            wait = 0
            max_valid_acc = max(max_valid_acc, valid_accs[-1])

        message = f"\nEnd of epoch: {epoch}, \
                \ntrain loss: {train_losses[-1]:.4f}, train top-1 acc: {train_accs[-1]:.4f}, \
                \nvalid loss: {valid_losses[-1]:.4f}, valid top-1 acc: {valid_accs[-1]:.4f}\n"
        logging.info(message)

    logging.info(f'Finished training, total time (minutes): {(time.time() - start) / 60}')
    return model

