import torch
from scipy import sparse
from torch.utils.data import Dataset


class FingerprintDataset(Dataset):
    """
    model agnostic dataset, relies on user to provide correct path_steps & path_states
    as training data (input & label) for each of the four models is slightly different
    """

    def __init__(self, path_steps, path_states):
        self.steps = sparse.load_npz(path_steps)
        self.states = sparse.load_npz(path_states)
        self.steps = self.steps.tocsr()
        self.states = self.states.tocsr()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        state = torch.as_tensor(self.states[idx].toarray()).squeeze(0)
        step = torch.as_tensor(self.steps[idx].toarray()).squeeze(0)
        # mask = torch.sum(step.bool(), axis=1).bool()

        return state.float(), step.float()

    def __len__(self):
        return self.states.shape[0]
