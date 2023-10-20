
import torch.nn as nn
from torch.utils.data import Dataset

import sys
sys.path.append('..')
import os
import pandas as pd
from utils import Gram, str2int, VOCAB_PROTEIN, VOCAB_LIGAND_ISO

class DTIDataset(Dataset):
    def __init__(self, root_dir, csv_name, embed_prot=len(VOCAB_PROTEIN), embed_lig=len(VOCAB_LIGAND_ISO), is_Gram = False):
        super(DTIDataset).__init__()
        self.embed_lig = nn.Embedding(embed_lig, 16, padding_idx=0)
        self.embed_prot = nn.Embedding(embed_prot, 16, padding_idx=0)
        self.is_Gram = is_Gram
        self.path = os.path.join(root_dir, csv_name)
        self.data = pd.read_csv(self.path)
        self.embed_drug = nn.Embedding(embed_lig, embed_lig, padding_idx=0)
        self.embed_target = nn.Embedding(embed_prot, embed_prot, padding_idx=0)

    def __getitem__(self, idx):
        ligand , protein, label =  self.data.iloc[idx,]['smiles'], self.data.iloc[idx,]['sequence'], self.data.iloc[idx,]['label']
        ligand, protein, label = str2int(ligand, protein, label)
        if self.is_Gram:
            ligand = Gram(self.embed_lig(ligand))
            protein = Gram(self.embed_prot(protein))
        return ligand, protein, label
    
    def __len__(self):
        return len(self.data)