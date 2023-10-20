import torch
import torch.nn as nn
import sys
sys.path.append('..')
from network import Conv1d, QuLinear
from utils import *

class QuDeepDTA(nn.Module):
    """DeepDTA model architecture, Y-shaped net that does 1d convolution on 
    both the ligand and the protein representation and then concatenates the
    result into a final predictor of binding affinity"""

    def __init__(self, pro_vocab_size, lig_vocab_size, channel, protein_kernel_size, ligand_kernel_size):
        super(QuDeepDTA, self).__init__()
        self.ligand_conv = Conv1d(lig_vocab_size, channel, ligand_kernel_size)
        self.protein_conv = Conv1d(pro_vocab_size, channel, protein_kernel_size)
        self.affinity = QuLinear(5)

    def forward(self, protein, ligand):
        x1 = self.ligand_conv(ligand)
        x2 = self.protein_conv(protein)
        x = torch.cat((x1, x2), dim=1).reshape(-1,6,32)
        aff = []
        for i in range(x.shape[0]):
            a = torch.squeeze(x[i])
            a = a.T@a
            a = encoding(a)
            aff.append(self.affinity(a))
        x = torch.tensor(aff, requires_grad=True)
        return x
