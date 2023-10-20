'''
Author: Dandan Liu
Date: 2023-09-27 15:59:25
'''
import torch
import torch.nn as nn
import sys
sys.path.append('..')
from network import Conv1d

class DeepDTA(nn.Module):
    """DeepDTA model architecture, Y-shaped net that does 1d convolution on 
    both the ligand and the protein representation and then concatenates the
    result into a final predictor of binding affinity"""

    def __init__(self, pro_vocab_size, lig_vocab_size, channel, protein_kernel_size, ligand_kernel_size):
        super(DeepDTA, self).__init__()
        self.ligand_conv = Conv1d(lig_vocab_size, channel, ligand_kernel_size)
        self.protein_conv = Conv1d(pro_vocab_size, channel, protein_kernel_size)
        self.fc1 = nn.Linear(channel*6, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.affinity = nn.Sequential(
            nn.Linear(channel*6, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.Dropout(0.1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 1),
        )
        self.fc4 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, protein, ligand):
        x1 = self.ligand_conv(ligand)
        x2 = self.protein_conv(protein)
        x = torch.cat((x1, x2), dim=1)
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.fc3(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.fc4(x)
        x = self.affinity(x)
        return x
