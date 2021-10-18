import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import AllChem, rdChemReactions
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

VALID_NUM_RCTS = set([1, 2])

def is_valid_reactant(mol_or_smi, template_str):

    # check type
    if isinstance(mol_or_smi, str):
        mol = Chem.MolFromSmiles(mol_or_smi)
    elif isinstance(mol_or_smi, Chem.Mol):
        mol = mol_or_smi
    else:
        raise TypeError(f'invalid input molecule type: {type(mol_or_smi)}')

    # TODO: this conversion step can be cut if we create a Template class
    # that holds both template_str & rdkit Reaction (and potentially other info) as its attributes
    rxn = AllChem.ReactionFromSmarts(template_str)
    rdChemReactions.ChemicalReaction.Initialize(rxn)

    num_rcts = rxn.GetNumReactantTemplates() # len(template_str.split('>')[0].split('.'))
    if num_rcts not in VALID_NUM_RCTS:
        raise ValueError(f'invalid num_rcts {num_rcts}')

    valid_rt1, valid_rt2 = False, False

    # check if molecule matches subgraph pattern of reactant #1 in current template
    rct1_template_str = template_str.split('>')[0].split('.')[0]
    if mol.HasSubstructMatch(Chem.MolFromSmarts(rct1_template_str)):
        valid_rt1 = True

    # bi-molecular reaction template
    if num_rcts == 2:
        # check if molecule matches subgraph pattern of reactant #2 in current template
        rct2_template_str = template_str.split('>')[0].split('.')[1]
        if mol.HasSubstructMatch(Chem.MolFromSmarts(rct2_template_str)):
            valid_rt2 = True

    return valid_rt1, valid_rt2

def smi_to_bit_fp(smi, radius=2, fp_size=4096):
    mol = Chem.MolFromSmiles(smi)

    fp_gen = GetMorganGenerator(
        radius=radius, useCountSimulation=False,
        includeChirality=False, fpSize=fp_size
    )
    uint_bit_fp = fp_gen.GetFingerprint(mol)
    bit_fp = np.empty((1, fp_size), dtype="int32")
    DataStructs.ConvertToNumpyArray(uint_bit_fp, bit_fp)

    return bit_fp # sparse.csr_matrix(bit_fp, dtype="int32")



def seed_everything(seed):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True