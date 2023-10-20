import torch
import numpy as np
import math
from scipy.linalg import sqrtm,logm

np.seterr(divide='ignore',invalid='ignore')

VOCAB_PROTEIN = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                 "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                 "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                 "U": 19, "T": 20, "W": 21,
                 "V": 22, "Y": 23, "X": 24,
                 "Z": 25}
VOCAB_LIGAND_ISO = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                    "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                    "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                    "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                    "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                    "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                    "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                    "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

def str2int(ligand, protein, label):
    ligand = [VOCAB_LIGAND_ISO[s] for s in ligand]
    protein = [VOCAB_PROTEIN[s] for s in protein]
        
    if len(ligand) < 128:
         ligand = np.pad(ligand, (0, 128 - len(ligand)))
    else:
        ligand = ligand[:128]
    if len(protein) < 512:
        protein = np.pad(protein, (0, 512 - len(protein)))
    else:
        protein = protein[:512]

    return torch.tensor(ligand, dtype=torch.long), torch.tensor(protein, dtype=torch.long), torch.tensor(label, dtype=torch.float)

def Gram(data, hyber_para = 16):
    data = data.T @ data
    data = data.view(hyber_para, hyber_para)
    return data

def get_mse(records_real, records_predict):
    """
    均方误差 估计值与真值 偏差
    """
    if len(records_real) == len(records_predict):
        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None

def get_rmse(records_real, records_predict):
    """
    均方根误差：是均方误差的算术平方根
    """
    mse = get_mse(records_real, records_predict)
    if mse:
        return math.sqrt(mse)
    else:
        return None
    
def cal_pccs(x, y, n):
    """
    warning: data format must be narray
    :param x: Variable 1
    :param y: The variable 2
    :param n: The number of elements in x
    :return: pccs
    """
    sum_xy = np.sum(np.sum(x*y))
    sum_x = np.sum(np.sum(x))
    sum_y = np.sum(np.sum(y))
    sum_x2 = np.sum(np.sum(x*x))
    sum_y2 = np.sum(np.sum(y*y))
    pcc = (n*sum_xy-sum_x*sum_y)/np.sqrt((n*sum_x2-sum_x*sum_x)*(n*sum_y2-sum_y*sum_y))
    return pcc

# 量子线路相关的函数
def dag(x):
    """
    compute conjugate transpose of input matrix
    """
    x_conj = torch.conj(x)
    x_dag = x_conj.permute(1, 0)
    return x_dag

def encoding(x):
    """
    input: n*n matrix
    perform L2 regularization on x, x为complex
    """

    with torch.no_grad():
        # x = x.squeeze( )
        if x.norm() != 1:
            xd = x.diag()
            xds = (xd.sqrt()).unsqueeze(1)
            xdsn = xds / (xds.norm() + 1e-12)
            xdsn2 = xdsn @ dag(xdsn)                    #dag() 自定义函数
            xdsn2 = xdsn2.type(dtype=torch.complex64)
        else:
            xdsn2 = x.type(dtype=torch.complex64)
    return xdsn2

def rx(phi):
    """Single-qubit rotation for operator sigmax with angle phi.
    -------
    result : torch.tensor for operator describing the rotation.
    """

    return torch.cat((torch.cos(phi / 2).unsqueeze(dim = 0), -1j * torch.sin(phi / 2).unsqueeze(dim = 0), 
                         -1j * torch.sin(phi / 2).unsqueeze(dim = 0), torch.cos(phi / 2).unsqueeze(dim = 0)),dim = 0).reshape(2,-1)
    # return torch.tensor([[torch.cos(phi / 2), -1j * torch.sin(phi / 2)],
    #              [-1j * torch.sin(phi / 2), torch.cos(phi / 2)]])

def ry(phi):
    """Single-qubit rotation for operator sigmay with angle phi.
    -------
    result : torch.tensor for operator describing the rotation.
    """

    return torch.cat((torch.cos(phi / 2).unsqueeze(dim = 0), -1 * torch.sin(phi / 2).unsqueeze(dim = 0), 
                      torch.sin(phi / 2).unsqueeze(dim = 0), torch.cos(phi / 2).unsqueeze(dim = 0)), dim = 0).reshape(2,-1) + 0j
    # return torch.tensor([[torch.cos(phi / 2), -torch.sin(phi / 2)],
    #              [torch.sin(phi / 2), torch.cos(phi / 2)]])

def rz(phi):
    """Single-qubit rotation for operator sigmaz with angle phi.
    -------
    result : torch.tensor for operator describing the rotation.
    """
    return torch.cat((torch.exp(-1j * phi / 2).unsqueeze(dim = 0), torch.zeros(1), 
                      torch.zeros(1), torch.exp(1j * phi / 2).unsqueeze(dim = 0)), dim = 0).reshape(2,-1)    
    # return torch.tensor([[torch.exp(-1j * phi / 2), 0],
    #              [0, torch.exp(1j * phi / 2)]])

def z_gate():
    """
    Pauli z
    """
    return torch.tensor([[1, 0], [0, -1]]) + 0j

def cnot():
    """
    torch.tensor representing the CNOT gate.
    control=0, target=1
    """
    return torch.tensor([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]]) + 0j

def Hcz():
    """
    controlled z gate for measurement
    """
    return torch.tensor([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, -1]]) + 0j  

def rxx(phi):
    """
    torch.tensor representing the rxx gate with angle phi.
    """
    return torch.kron(rx(phi), rx(phi))

def ryy(phi):
    """
    torch.tensor representing the ryy gate with angle phi.
    """
    return torch.kron(ry(phi), ry(phi))

def rzz(phi):
    """
    torch.tensor representing the rzz gate with angle phi.
    """
    return torch.kron(rz(phi), rz(phi))

def multi_kron(x_list):
    """
    kron the data in the list in order
    """
    x_k = torch.ones(1)
    for x in x_list:
        x_k = torch.kron(x_k, x)
    return x_k

def gate_expand_1toN(U, N, target):
    """
    representing a one-qubit gate that act on a system with N qubits.

    """

    if N < 1:
        raise ValueError("integer N must be larger or equal to 1")

    if target >= N:
        raise ValueError("target must be integer < integer N")

    return multi_kron([torch.eye(2)]* target + [U] + [torch.eye(2)] * (N - target - 1))

def gate_expand_2toN(U, N, targets):
    """
    representing a two-qubit gate that act on a system with N qubits.
    
    """

    if N < 2:
        raise ValueError("integer N must be larger or equal to 2")

    if targets[1] >= N:
        raise ValueError("target must be integer < integer N")

    return multi_kron([torch.eye(2)]* targets[0] + [U] + [torch.eye(2)] * (N - targets[1] - 1))

def gate_sequence_product(U_list, n_qubits, left_to_right=True):
    """
    Calculate the overall unitary matrix for a given list of unitary operations.
    return: Unitary matrix corresponding to U_list.
    """

    U_overall = torch.eye(2 ** n_qubits, 2 **  n_qubits) + 0j
    for U in U_list:
        if left_to_right:
            U_overall = U @ U_overall
        else:
            U_overall = U_overall @ U

    return U_overall

def ptrace(rhoAB, dimA, dimB):
    """
    rhoAB : density matrix（密度矩阵）
    dimA: n_qubits A keep
    dimB: n_qubits B trash
    """
    mat_dim_A = 2**dimA
    mat_dim_B = 2**dimB

    id1 = torch.eye(mat_dim_A, requires_grad=True) + 0.j
    id2 = torch.eye(mat_dim_B, requires_grad=True) + 0.j

    pout = 0
    for i in range(mat_dim_B):
        p = torch.kron(id1, id2[i]) @ rhoAB @ torch.kron(id1, id2[i].reshape(mat_dim_B, 1))
        pout += p
    return pout

def expecval_ZI(state, nqubit, target):
    """
    state为nqubit大小的密度矩阵，target为z门放置位置
    
    """
    zgate=z_gate()
    H = gate_expand_1toN(zgate, nqubit, target)   #gate_expand_1toN（） 为自定义函数
    expecval = (state @ H).trace() #[-1,1]
    expecval_real = (expecval.real + 1) / 2 #[0,1]
    
    return expecval_real

def measure_affinity(state, nqubit, site=0):
    measure = torch.zeros(nqubit, 1)
    measure = expecval_ZI(state, nqubit, site)
    return measure
