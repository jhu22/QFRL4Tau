import numpy as np
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

inputsize = 16384   #16384 256
middlesize = 500
outputsize = 1
epsilon = 0.2

file1 = np.load('file1.npy', allow_pickle=True).item()
file2 = np.load('file2.npy', allow_pickle=True).item()
file3 = np.load('file3.npy').tolist()
Deadend = np.load('Deadend.npy').tolist()
buyable = np.load('buyable.npy').tolist()


class Model(nn.Module):
    def __init__(self, inputsize, middlesize, outputsize):
        super(Model, self).__init__()
        self.relu = nn.ReLU()
        self.value_fc1 = nn.Linear(in_features=inputsize, out_features=middlesize)
        self.value_fc2 = nn.Linear(in_features=middlesize, out_features=outputsize)

    def forward(self, state):
        v = self.relu(self.value_fc1(state))
        v = self.value_fc2(v)
        return v


class Tree:
    def __init__(self):
        self.depth = 1
        self.maxdepth = 10
        self.cost1 = {}
        self.cost2 = {}
        self.NN = Model(inputsize, middlesize, outputsize)
        self.is_model = False
        self.layer = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
        self.updates = 0
        self.epsilon = 0.2
        self.loss_fn = torch.nn.MSELoss()
        self.opt = torch.optim.SGD(self.NN.parameters(), lr= 0.01/(1+2*math.sqrt(self.updates)))

        self.tocost = {}
        self.avtocost = []
        self.lossv = []

    def game(self):
        for episode in range(1, 201):
            episodecost = 0
            for name in file3:
                self.layer[1] = [name]
                # namecost = 0
                while self.depth < 10:
                    if self.layer[self.depth] == []:
                        # print(name)
                        self.depth = 10
                    else:
                        for name in self.layer[self.depth]:
                            rm, minv = self.choosereaction(name)
                            # namecost += minv
                            if self.depth == 1:
                                episodecost += minv
                            if (name, 10 - self.depth) in self.cost2.keys():
                                if len(self.cost2[name, 10 - self.depth]) >= 100:
                                    self.cost2[name, 10 - self.depth].pop(0)
                                    self.cost2[name, 10 - self.depth].append(minv)
                                else:
                                    self.cost2[name, 10 - self.depth].append(minv)
                            else:
                                self.cost2[name, 10 - self.depth] = [minv]
                            if rm:
                                for m in rm:
                                    self.add_child(m)
                        self.depth += 1
                # if name in self.tocost.keys():
                #     self.tocost[name].append(namecost)
                # else:
                #     self.tocost[name] = [namecost]
                self.merge()
                self.renew()
            with torch.no_grad():
                avc = float(episodecost)/len(file3)
            self.avtocost.append(avc)
            self.merge()

            if episode > 20:
                if episode % 5 == 0:
                    self.updates += 1
                    self.is_model = True
                    self.train()
                if episode % 10 == 0:
                    if self.epsilon-0.08 > 0:
                        self.epsilon -= 0.08
                    else:
                        self.epsilon = 0
                    self.cost1.clear()
                    self.cost2.clear()
            print(episode)

    def renew(self):
        self.depth = 1
        self.layer = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}

    def choosereaction(self, name):
        minv = sys.maxsize
        mink = None
        if name not in file1.keys():
            if name in buyable:
                rmv = 0
            else:
                rmv = 100
            return None, rmv
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
                                if (m, 9 - self.depth) in self.cost2.keys():
                                    self.cost2[m, 9 - self.depth].append(rmv)
                                else:
                                    self.cost2[m, 9 - self.depth] = [rmv]
                            else:
                                rmv = np.random.randint(1, 101)
                            # if (m, 9 - self.depth) in self.cost2.keys():
                            #     self.cost2[m, 9 - self.depth].append(rmv)
                            # else:
                            #     self.cost2[m, 9 - self.depth] = [rmv]
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
                                if (m, 9 - self.depth) in self.cost2.keys():
                                    self.cost2[m, 9 - self.depth].append(rmv)
                                else:
                                    self.cost2[m, 9 - self.depth] = [rmv]
                            else:
                                fp = torch.tensor(file2[m], dtype=torch.float)
                                depth = torch.tensor([9 - self.depth], dtype=torch.float)
                                fp = torch.cat([fp, depth])
                                rmv = self.NN.forward(fp)
                                if rmv < -1000:
                                    rmv = 0
                            # if (m, 9 - self.depth) in self.cost2.keys():
                            #     self.cost2[m, 9 - self.depth].append(rmv)
                            # else:
                            #     self.cost2[m, 9 - self.depth] = [rmv]
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
                            if (m, 9 - self.depth) in self.cost2.keys():
                                self.cost2[m, 9 - self.depth].append(rmv)
                            else:
                                self.cost2[m, 9 - self.depth] = [rmv]
                        else:
                            rmv = np.random.randint(1, 101)
                        # if (m, 9 - self.depth) in self.cost2.keys():
                        #     self.cost2[m, 9 - self.depth].append(rmv)
                        # else:
                        #     self.cost2[m, 9 - self.depth] = [rmv]
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
                            if (m, 9 - self.depth) in self.cost2.keys():
                                self.cost2[m, 9 - self.depth].append(rmv)
                            else:
                                self.cost2[m, 9 - self.depth] = [rmv]
                        else:
                            fp = torch.tensor(file2[m], dtype=torch.float)
                            depth = torch.tensor([9 - self.depth], dtype=torch.float)
                            fp = torch.cat([fp, depth])
                            rmv = self.NN.forward(fp)
                            if rmv < -1000:
                                rmv = 0
                        # if (m, 9 - self.depth) in self.cost2.keys():
                        #     self.cost2[m, 9 - self.depth].append(rmv)
                        # else:
                        #     self.cost2[m, 9 - self.depth] = [rmv]
                        tempv = tempv + rmv
            minv = tempv

        return file1[name][mink], minv

    def add_child(self, m):
        self.layer[self.depth + 1].append(m)

    def merge(self):
        self.cost1.update(self.cost2)
        self.cost2.clear()

    def train(self):
        for epoch in range(50):
            # print('epoch', epoch)
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
            x = torch.tensor(x, dtype=torch.float)
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
                    if self.depth == 1:
                        namecost += minv
                    if rm:
                        for m in rm:
                            self.add_child(m)
                self.depth += 1
        return namecost

    def valuem(self, m, depth):
        # 1 2 3 4 5 6 7 8 9 10
        #→9 8 7 6 5 4 3 2 1 0
        fp = torch.tensor(file2[m], dtype=torch.float)
        depth = torch.tensor([depth], dtype=torch.float)
        fp = torch.cat([fp, depth])
        rmv = self.NN.forward(fp)
        return rmv


# tree =Tree()
# tree.game()

# np.save('tocost.npy', tree.tocost)
# np.save('lossv.npy', tree.lossv)


# import matplotlib.pyplot as plt
# lossv = np.load('lossv.npy').tolist()
# plt.plot(range(0,2000),lossv,label='lossv')

# import matplotlib.pyplot as plt
# plt.plot(range(0,len(tree.avtocost)),tree.avtocost,label='avtocost')
# plt.plot(range(0,len(tree.lossv)),tree.lossv,label='lossv')
# plt.savefig('avtocostxx.svg',dpi=300,format='svg')
