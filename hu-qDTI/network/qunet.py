import torch
from torch import nn
import numpy as np
import sys
sys.path.append('..')

from utils import *

class QuLinear(nn.Module):
    """
    根据量子线路图摆放旋转门以及受控门
    """

    def __init__(self, n_qubits, gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()

        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(30), a=0.0, b=2*np.pi) * init_std)
        
        self.n_qubits = n_qubits


    def layer(self):
        w = self.weight * self.w_mul
        cir = []
        
        #旋转门
        for which_q in range(0, self.n_qubits):
            #print(which_q)
            cir.append(gate_expand_1toN(rx(w[which_q]), self.n_qubits, which_q))
            cir.append(gate_expand_1toN(ry(w[which_q+10]), self.n_qubits, which_q ))
            cir.append(gate_expand_1toN(rz(w[which_q+20]), self.n_qubits, which_q))
            
        
        #cnot门
        for which_q in range(1,self.n_qubits):
            #print(which_q)
            cir.append(gate_expand_2toN(cnot(), self.n_qubits, [which_q-1, which_q]))
            
            
        #旋转门
        for which_q in range(0, self.n_qubits):
            #print(which_q)
            cir.append(gate_expand_1toN(rx(-w[which_q]), self.n_qubits, which_q))
            cir.append(gate_expand_1toN(ry(-w[which_q+10]), self.n_qubits, which_q ))
            cir.append(gate_expand_1toN(rz(-w[which_q+20]), self.n_qubits, which_q))
        
        for which_q in range(1,self.n_qubits):
            cir.append(gate_expand_2toN(cnot(), self.n_qubits, [which_q-1, which_q]))


        U = gate_sequence_product(cir, self.n_qubits)        
        return U
    
    def forward(self, x):
        # 得到酉矩阵
        E_qlayer = self.layer()
        # 酉矩阵和量子态数据相乘
        x =  E_qlayer @ x @ dag(E_qlayer)
        # 测量
        output = measure_affinity(x, 5)
        return output
