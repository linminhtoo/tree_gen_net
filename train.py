import argparse
import logging
import os
import sys
import time
from pathlib import Path

import nmslib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_dir)
from data_scripts.utils import knn_search_batch, seed_everything, numel
from dataset.dataset import FingerprintDataset
from model.basic import BasicFeedforward


def train(args):
    seed_everything(args.seed)

    print("creating checkpoint folder")
    (args.path_checkpoint).mkdir(parents=True, exist_ok=args.checkpoint_existok)

    # load templates
    with open(args.path_templates, "r") as f:
        template_strs = [l.strip().split("|")[1] for l in f.readlines()]

    # load kNN search index (nmslib)
    index_all_mols = nmslib.init(method="hnsw", space="cosinesimil")
    index_all_mols.loadIndex(str(args.path_index), load_data=True)

    # instantiate model & loss function
    hidden_sizes = list(map(int, args.hidden_sizes.split(",")))
    if args.model == "f_act":
        model = BasicFeedforward(
            input_size=args.input_fp_dim * 3,
            act_fn="ReLU",
            hidden_sizes=hidden_sizes,  # [1000, 1200, 3000, 3000]
            output_size=4,
            dropout=args.dropout,  # 0
            final_act_fn="softmax",
        )
        criterion = nn.CrossEntropyLoss(reduction="sum")

    elif args.model == "f_rxn":
        model = BasicFeedforward(
            input_size=args.input_fp_dim * 4,
            act_fn="ReLU",
            hidden_sizes=hidden_sizes,  # [1000, 1200, 3000, 3000]
            output_size=len(template_strs),
            dropout=args.dropout,  # 0
            final_act_fn="softmax",
        )
        criterion = nn.CrossEntropyLoss(reduction="sum")

    elif args.model == "f_rt1":
        model = BasicFeedforward(
            input_size=args.input_fp_dim * 3,
            act_fn="ReLU",
            hidden_sizes=hidden_sizes,  # [1000, 1200, 3000, 3000]
            output_size=args.output_fp_dim,
            dropout=args.dropout,  # 0
            final_act_fn=None,
        )
        criterion = nn.MSELoss(reduction="sum")

    elif args.model == "f_rt2":
        model = BasicFeedforward(
            input_size=args.input_fp_dim * 4 + len(template_strs),
            act_fn="ReLU",
            hidden_sizes=hidden_sizes,  # [1000, 1200, 3000, 3000]
            output_size=args.output_fp_dim,
            dropout=args.dropout,  # 0
            final_act_fn=None,
        )
        criterion = nn.MSELoss(reduction="sum")

    else:
        raise ValueError(f"inval model name {args.model}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    model = model.to(device)
    print(f"number of parameters in model: {numel(model)}")

    if args.optim == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f"unrecognized optimizer {args.optim}")

    # prepare train & val datasets
    train_dataset = FingerprintDataset(args.path_steps_train, args.path_states_train)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)

    val_dataset = FingerprintDataset(args.path_steps_val, args.path_states_val)
    val_loader = DataLoader(val_dataset, batch_size=args.bs_eval, shuffle=False)
    del train_dataset, val_dataset

    # start training
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    max_val_acc = float("-inf")
    wait = 0  # early stopping patience counter

    start = time.time()
    for epoch in range(args.epochs):
        train_loss, train_correct, train_seen = 0, 0, 0
        train_loader = tqdm(train_loader, desc="training", disable=args.logging)
        model.train()

        for data in train_loader:
            X, y = data

            if args.model == "f_rt1" or args.model == "f_rt2":
                # split y into mol_embedding + mol_idx
                rct_idxs = y[:, -1].unsqueeze(-1)
                y = y[:, :-1]
            else:
                y = torch.argmax(y, dim=-1).long()
            X, y = X.to(device), y.to(device)

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
                pred_idxs = torch.argmax(y_pred, dim=-1)
                train_correct += torch.sum(torch.eq(pred_idxs, y), dim=0).item()

            else:
                # nearest neighbor search over all building block embeddings to get nearest idx
                # compare with groundtruth building block idx
                pred_idxs, _ = knn_search_batch(
                    y_pred.detach().cpu().numpy(), index_all_mols, k=1
                )
                pred_idxs = torch.Tensor(pred_idxs)

                train_correct += torch.sum(torch.eq(pred_idxs, rct_idxs), dim=0).item()

            if not args.logging:
                train_loader.set_description(
                    f"training: loss={train_loss / train_seen:.4f}, top-1 acc={train_correct / train_seen:.4f}"
                )
                train_loader.refresh()

        train_losses.append(train_loss / train_seen)
        train_accs.append(train_correct / train_seen)

        # validation
        model.eval()
        with torch.no_grad():
            val_loss, val_correct, val_seen = 0, 0, 0
            val_loader = tqdm(val_loader, desc="validating", disable=args.logging)

            for data in val_loader:
                X, y = data

                if args.model == "f_rt1" or args.model == "f_rt2":
                    # split y into mol_embedding + mol_idx
                    rct_idxs = y[:, -1].unsqueeze(-1)
                    y = y[:, :-1]
                else:
                    y = torch.argmax(y, dim=-1).long()
                X, y = X.to(device), y.to(device)

                y_pred = model(X)
                loss = criterion(y_pred, y)

                val_loss += loss.item()
                val_seen += y.shape[0]

                if args.model == "f_act" or args.model == "f_rxn":
                    # calculate accuracy
                    pred_idxs = torch.argmax(y_pred, dim=-1)
                    val_correct += torch.sum(torch.eq(pred_idxs, y), dim=0).item()

                else:
                    # nearest neighbor search over all building block embeddings to get nearest idx
                    # compare with groundtruth building block idx
                    pred_idxs, _ = knn_search_batch(
                        y_pred.detach().cpu().numpy(), index_all_mols, k=1
                    )
                    pred_idxs = torch.Tensor(pred_idxs)

                    val_correct += torch.sum(
                        torch.eq(pred_idxs, rct_idxs), dim=0
                    ).item()

                if not args.logging:
                    val_loader.set_description(
                        f"validating: loss={val_loss / val_seen:.4f}, top-1 acc={val_correct / val_seen:.4f}"
                    )
                    val_loader.refresh()

        val_losses.append(val_loss / val_seen)
        val_accs.append(val_correct / val_seen)

        if val_accs[-1] > max_val_acc:
            # save best model
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_accs": train_accs,
                    "train_losses": train_losses,
                    "val_accs": val_accs,
                    "val_losses": val_losses,
                    "max_val_acc": max_val_acc,
                },
                args.path_checkpoint / "best.pth.tar",
            )
            print("Saved best model")

        if args.early_stop and max_val_acc - val_accs[-1] > 0:
            if args.early_stop_patience <= wait:
                message = f"\nEarly stopped at the end of epoch: {epoch}, \
                \nval loss: {val_losses[-1]:.4f}, val top-1 acc: {val_accs[-1]:.4f} \
                \nbest val top-1 acc: {max_val_acc:.4f}\n"
                print(message)
                break
            else:
                wait += 1
                print(
                    f"\nvalidation accuracy did not increase, patience count: {wait}\n"
                )
        else:
            wait = 0
            max_val_acc = max(max_val_acc, val_accs[-1])

        message = f"\nEnd of epoch: {epoch}, \
                \ntrain loss: {train_losses[-1]:.4f}, train top-1 acc: {train_accs[-1]:.4f}, \
                \nval loss: {val_losses[-1]:.4f}, val top-1 acc: {val_accs[-1]:.4f} \
                \nbest val top-1 acc: {max_val_acc:.4f}\n"
        print(message)

    # save last checkpoint
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_accs": train_accs,
            "train_losses": train_losses,
            "val_accs": val_accs,
            "val_losses": val_losses,
            "max_val_acc": max_val_acc,
        },
        args.path_checkpoint / "last.pth.tar",
    )
    print("Saved last model")
    print(f"Finished training, total time (minutes): {(time.time() - start) / 60 :.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # input data arguments
    parser.add_argument(
        "--path_states_train", type=Path, default="data/states_f_act_train.npz"
    )
    parser.add_argument(
        "--path_states_val", type=Path, default="data/states_f_act_val.npz"
    )
    parser.add_argument(
        "--path_steps_train", type=Path, default="data/steps_f_act_train.npz"
    )
    parser.add_argument(
        "--path_steps_val", type=Path, default="data/steps_f_act_val.npz"
    )
    parser.add_argument("--path_index", type=Path, default="data/knn_rct_fps.index")
    parser.add_argument(
        "--path_templates", type=Path, default="data/templates_cleaned.txt"
    )
    # model arguments
    parser.add_argument(
        "--model",
        type=str,
        help="model type to train, among the four networks ['f_act', 'f_rxn', 'f_rt1', 'f_rt2']",
    )
    parser.add_argument("--bs", type=int, default=64, help="batch size, training")
    parser.add_argument(
        "--bs_eval", type=int, default=96, help="batch size, validation"
    )
    parser.add_argument(
        "--input_fp_dim",
        type=int,
        default=4096,
        help="dim of each input fingerprint into network",
    )
    parser.add_argument(
        "--output_fp_dim",
        type=int,
        default=256,
        help="dim of each output fingerprint from network",
    )
    parser.add_argument(
        "--hidden_sizes",
        type=str,
        default="1000,1200,3000,3000",
        help="network hidden_sizes, separated by commas",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout value")
    # training arguments
    parser.add_argument(
        "--path_checkpoint", type=Path, help="path to store model checkpoints"
    )
    parser.add_argument(
        "--checkpoint_existok",
        action="store_true",
        help="whether to override existing checkpoint",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="number of training epochs"
    )
    parser.add_argument(
        "--early_stop", action="store_true", help="whether to early stop"
    )
    parser.add_argument(
        "--logging", action="store_true", help="whether logging to disk (disables tqdm)"
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=10,
        help="early stop patience (number of epochs)",
    )
    parser.add_argument("--seed", type=int, default=1337, help="random seed")
    parser.add_argument("--optim", type=str, default="Adam", help="optimizer")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    args = parser.parse_args()

    print(args)
    train(args)
