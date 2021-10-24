from typing import List, Optional

import torch
import torch.nn as nn

Tensor = torch.Tensor


class BasicFeedforward(nn.Module):
    """
    implements a basic feedforward network
    which can be used to parameterize all of f_act, f_rt1, f_rxn and f_rt2 from the paper

    hidden_sizes : List[int]
        list of hidden layer sizes for the encoder, from layer 0 onwards
        e.g. [1024, 512, 256] = layer 0 has 1024 neurons, layer 1 has 512 neurons etc.
    output_size : List[int]
        how many outputs the model should give
    """

    def __init__(
        self,
        input_size=4096,
        act_fn="ReLU",
        hidden_sizes=[1000, 1200, 3000, 3000],
        output_size=256,
        dropout=0,
        final_act_fn=None,
    ):
        super().__init__()

        self.encoder = self.build(
            input_size, hidden_sizes, act_fn, dropout, is_last=False
        )
        self.output = self.build(
            hidden_sizes[-1], [output_size], None, None, is_last=True
        )
        if final_act_fn:
            if final_act_fn == "softmax":
                act = nn.Softmax(dim=-1)
            else:
                raise ValueError(f"final activation {final_act_fn} is unrecognized")
            self.final_act_fn = act
        else:
            self.final_act_fn = None

    def build(self, input_size, hidden_sizes, act_fn, dropout, is_last=False):
        if act_fn:
            if act_fn == "ReLU":
                act = nn.ReLU()
            else:
                raise ValueError(f"activation {act_fn} is unrecognized")

        num_layers = len(hidden_sizes)

        if is_last:
            block = [nn.Linear(input_size, hidden_sizes[0])]
        else:
            block = [
                nn.Linear(input_size, hidden_sizes[0]),
                nn.BatchNorm1d(num_features=hidden_sizes[0]),
                act,
            ]
            for i in range(num_layers - 1):
                block.extend(
                    [
                        nn.Dropout(dropout),
                        nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                        nn.BatchNorm1d(num_features=hidden_sizes[i + 1]),
                        act,
                    ]
                )
        return nn.Sequential(*block)

    def forward(self, batch, mask=None):
        output = self.output(self.encoder(batch))

        if mask:
            # mask out invalid actions or templates, before final softmax
            output = output * mask

        if self.final_act_fn:
            return self.final_act_fn(output)
        else:
            return output


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_templates", type=Path, default="data/templates_cleaned.txt"
    )
    args = parser.parse_args()

    # load templates
    with open(args.path_templates, "r") as f:
        template_strs = [l.strip().split("|")[1] for l in f.readlines()]

    # action selection network
    f_act = BasicFeedforward(
        input_size=4096 * 3,
        act_fn="ReLU",
        hidden_sizes=[1000, 1200, 3000, 3000],
        output_size=4,
        dropout=0,
        final_act_fn="softmax",
    )
    sample_batch = torch.randn((32, 4096 * 3))
    sample_out = f_act(sample_batch)
    print(f"output shape from f_act: {sample_out.shape}")
    print(f"sample row: {sample_out[0]}")

    # reactant1 prediction network
    f_rt1 = BasicFeedforward(
        input_size=4096 * 3,
        act_fn="ReLU",
        hidden_sizes=[1000, 1200, 3000, 3000],
        output_size=256,
        dropout=0,
        final_act_fn=None,  # linear activation
    )
    sample_batch = torch.randn((32, 4096 * 3))
    sample_out = f_rt1(sample_batch)
    print(f"output shape from f_rt1: {sample_out.shape}")
    print(f"sample row: {sample_out[0]}")

    # reaction selection network
    f_rxn = BasicFeedforward(
        input_size=4096 * 4,
        act_fn="ReLU",
        hidden_sizes=[1000, 1200, 3000, 3000],
        output_size=len(template_strs),
        dropout=0,
        final_act_fn="softmax",
    )
    sample_batch = torch.randn((32, 4096 * 4))
    sample_out = f_rxn(sample_batch)
    print(f"output shape from f_rxn: {sample_out.shape}")
    print(f"sample row: {sample_out[0]}")

    # reactant2 prediciton network
    f_rt2 = BasicFeedforward(
        input_size=4096 * 4 + len(template_strs),
        act_fn="ReLU",
        hidden_sizes=[1000, 1200, 3000, 3000],
        output_size=256,
        dropout=0,
        final_act_fn=None,  # linear activation
    )
    sample_batch = torch.randn((32, 4096 * 4 + len(template_strs)))
    sample_out = f_rt2(sample_batch)
    print(f"output shape from f_rt2: {sample_out.shape}")
    print(f"sample row: {sample_out[0]}")
