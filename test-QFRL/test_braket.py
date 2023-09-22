import torch
import torch.nn as nn
import numpy as np
import random
import sys
import torch
import torch.nn.functional as F
from braket.aws import AwsDevice
import math
import time
from braket.circuits import Circuit

inputsize = 256
middlesize = 500
outputsize = 1
epsilon = 0.2

file1 = np.load('file1.npy', allow_pickle=True).item()
file2 = np.load('file2-255.npy', allow_pickle=True).item()
file3 = np.load('file3.npy').tolist()
Deadend = np.load('Deadend.npy').tolist()
buyable = np.load('buyable.npy').tolist()


class CirModel(nn.Module):

    def __init__(self, n_qubits):
        super().__init__()
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(6*n_qubits), a=0.0, b=2 * np.pi))
        self.n_qubits = n_qubits

    def forward(self, x):
        cir1 = Circuit()
        cir1.z(0)
        for i in range(1, self.n_qubits):
            cir1.i(i)
        M = torch.tensor(cir1.to_unitary()).type(dtype=torch.complex64)

        w = self.weight
        cir2 = Circuit()
        for which_q in range(0, self.n_qubits):
            cir2.ry(which_q, w[0+6*which_q])
            cir2.rz(which_q, w[1+6*which_q])
            cir2.ry(which_q, w[2+6*which_q])
            if which_q < (self.n_qubits-1):
                cir2.cnot(which_q, which_q + 1)
            else:
                cir2.cnot(which_q, 0)
            cir2.ry(which_q, w[3+6*which_q])
            cir2.rz(which_q, w[4+6*which_q])
            cir2.ry(which_q, w[5+6*which_q])
        unitary = torch.tensor(cir2.to_unitary(), requires_grad = True).type(dtype=torch.complex64)

        if x.shape[0] == 1:
            out = unitary @ x.T
            res = (out.conj().T @ M @ out).real
        else:
            out = unitary @ x.T
            res = (out.conj().T @ M @ out).diag().real
            res = res.reshape(-1,1)
        return res

class Tree:
    def __init__(self):
        self.depth = 1
        self.maxdepth = 10
        self.cost1 = {}
        self.cost2 = {}
        # self.NN = Model(inputsize, middlesize, outputsize)
        self.NN = CirModel(8)
        self.is_model = False
        self.layer = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
        self.updates = 0
        self.epsilon = 0.2
        self.loss_fn = torch.nn.MSELoss()
        self.opt = torch.optim.SGD(self.NN.parameters(), lr= 0.001/(1+2*math.sqrt(self.updates)))

        self.tocost = {}
        self.avtocost = []
        self.lossv = []

    def game(self):
        for episode in range(1, 301):
            episodecost = 0
            for name in file3:
                self.layer[1] = [name]
                namecost = 0
                while self.depth < 10:
                    if self.layer[self.depth] == []:
                        # print(name)
                        self.depth = 10
                    else:
                        for name in self.layer[self.depth]:
                            rm, minv = self.choosereaction(name)
                            namecost += minv
                            episodecost += minv
                            if rm:
                                for m in rm:
                                    self.add_child(m)
                        self.depth += 1
                # if name in self.tocost.keys():
                #     self.tocost[name].append(namecost)
                # else:
                #     self.tocost[name] = [namecost]
                self.renew()
            with torch.no_grad():
                avc = float(episodecost)/len(file3)
            self.avtocost.append(avc)
            self.merge()
            if episode % 30 == 0:
                self.updates += 1
                self.is_model = True
                self.train()
            if episode % 100 == 0:
                if self.epsilon-0.05 > 0:
                    self.epsilon -= 0.03
                else:
                    self.epsilon = 0
                self.cost1 = {}
                self.cost2 = {}
            print(episode)

    def renew(self):
        self.depth = 1
        self.layer = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}

    def choosereaction(self, name):
        minv = sys.maxsize
        mink = None
        if name not in file1.keys():
            return None, 0
        if np.random.random() > self.epsilon:  # name产物， r反应， rm反应物list， m反应物
            for r in file1[name].keys():
                rm = file1[name][r]
                tempv = 1
                if not self.is_model:
                    for m in rm:
                        if (m, 9 - self.depth) in self.cost1.keys():
                            rmv = sum(self.cost1[m, 9 - self.depth]) / len(self.cost1[m, 9 - self.depth])
                            tempv = tempv + rmv
                        else:
                            if m in buyable:
                                rmv = 0
                            elif m in Deadend:
                                rmv = 100
                            elif self.depth == 9:
                                rmv = 10
                            else:
                                rmv = np.random.randint(1, 101)
                            if (m, 9 - self.depth) in self.cost2.keys():
                                self.cost2[m, 9 - self.depth].append(rmv)
                            else:
                                self.cost2[m, 9 - self.depth] = [rmv]
                            tempv = tempv + rmv
                else:
                    for m in rm:
                        if (m, 9 - self.depth) in self.cost1.keys():
                            rmv = sum(self.cost1[m, 9 - self.depth]) / len(self.cost1[m, 9 - self.depth])
                            tempv = tempv + rmv
                        else:
                            if m in buyable:
                                rmv = 0
                            elif m in Deadend:
                                rmv = 100
                            elif self.depth == 9:
                                rmv = 10
                            else:
                                fp = torch.tensor(file2[m], dtype=torch.float)
                                depth = torch.tensor([self.depth], dtype=torch.float)
                                fp = torch.cat([fp, depth])
                                fp = fp.reshape(1,-1)+0j
                                fp = nn.functional.normalize(fp)
                                rmv = self.NN.forward(fp)[0]
                            if (m, 9 - self.depth) in self.cost2.keys():
                                self.cost2[m, 9 - self.depth].append(rmv)
                            else:
                                self.cost2[m, 9 - self.depth] = [rmv]
                            tempv = tempv + rmv
                if tempv < minv:
                    minv = tempv
                    mink = r
        else:
            mink = random.sample(file1[name].keys(), 1)[0]
            rm = file1[name][mink]
            tempv = 1
            if not self.is_model:
                for m in rm:
                    if (m, 9 - self.depth) in self.cost1.keys():
                        rmv = sum(self.cost1[m, 9 - self.depth]) / len(self.cost1[m, 9 - self.depth])
                        tempv = tempv + rmv
                    else:
                        if m in buyable:
                            rmv = 0
                        elif m in Deadend:
                            rmv = 100
                        elif self.depth == 9:
                            rmv = 10
                        else:
                            rmv = np.random.randint(1, 101)
                        if (m, 9 - self.depth) in self.cost2.keys():
                            self.cost2[m, 9 - self.depth].append(rmv)
                        else:
                            self.cost2[m, 9 - self.depth] = [rmv]
                        tempv = tempv + rmv
            else:
                for m in rm:
                    if (m, 9 - self.depth) in self.cost1.keys():
                        rmv = sum(self.cost1[m, 9 - self.depth]) / len(self.cost1[m, 9 - self.depth])
                        tempv = tempv + rmv
                    else:
                        if m in buyable:
                            rmv = 0
                        elif m in Deadend:
                            rmv = 100
                        elif self.depth == 9:
                            rmv = 10
                        else:
                            fp = torch.tensor(file2[m], dtype=torch.float)
                            depth = torch.tensor([self.depth], dtype=torch.float)
                            fp = torch.cat([fp, depth])
                            fp = fp.reshape(1, -1) + 0j
                            fp = nn.functional.normalize(fp)
                            rmv = self.NN.forward(fp)[0]
                        if (m, 9 - self.depth) in self.cost2.keys():
                            self.cost2[m, 9 - self.depth].append(rmv)
                        else:
                            self.cost2[m, 9 - self.depth] = [rmv]
                        tempv = tempv + rmv
            minv = tempv

        return file1[name][mink], minv

    def add_child(self, m):
        self.layer[self.depth + 1].append(m)

    def merge(self):
        self.cost1.update(self.cost2)

    def train(self):
        for epoch in range(100):
            print('epoch', epoch)
            y = []
            x = []
            temp = random.sample(self.cost1.keys(), 128)
            for i in temp:
                y.append(self.cost1[i][0])
                fp = np.array(file2[i[0]])
                depth = i[1]
                fp = np.append(fp, depth)
                x.append(fp)
            x = np.array(x)
            y = torch.tensor(y, dtype=torch.float).reshape(-1, 1)
            x = torch.tensor(x, dtype=torch.float)+0j
            x = nn.functional.normalize(x)
            # x = x.reshape(128,1,-1)
            output = self.NN.forward(x)
            loss = self.loss_fn(output, y)
            self.lossv.append(loss.data)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

    def pathway(self, name):
        self.renew()
        self.epsilon = 0
        self.layer[1] = [name]
        namecost = 0
        while self.depth < 10:
            if self.layer[self.depth] == []:
                # print(name)
                self.depth = 10
            else:
                for name in self.layer[self.depth]:
                    rm, minv = self.choosereaction(name)
                    namecost += minv
                    if rm:
                        for m in rm:
                            self.add_child(m)
                self.depth += 1
        return namecost


# tree =Tree()
# tree.game()