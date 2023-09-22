import torch
import torch.nn as nn
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from braket.aws import AwsDevice
import math
import time
from braket.circuits import Circuit
from torch.autograd import Function
import copy
import torch.optim as optim


class CirModel(nn.Module):

    def __init__(self, n_qubits, ):
        super().__init__()
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(6*n_qubits), a=0.0, b=2 * np.pi))
        self.n_qubits = n_qubits
        self.shift = torch.pi / 2

    def forward(self, x):
        res = HybridFunction.apply(self.weight, x, self.n_qubits, self.shift)

        return res

class qcircuit:
    def __init__(self, params, qubits):
        self.w = params
        self.qubits = qubits

    def run(self):
        cir2 = Circuit()
        for which_q in range(0, self.qubits):
            cir2.ry(which_q, self.w[0+6*which_q])
            cir2.rz(which_q, self.w[1+6*which_q])
            cir2.ry(which_q, self.w[2+6*which_q])
            if which_q < (self.qubits-1):
                cir2.cnot(which_q, which_q + 1)
            else:
                cir2.cnot(which_q, 0)
            cir2.ry(which_q, self.w[3+6*which_q])
            cir2.rz(which_q, self.w[4+6*which_q])
            cir2.ry(which_q, self.w[5+6*which_q])

        unitary = torch.tensor(cir2.to_unitary(), requires_grad = True).type(dtype=torch.complex64)
        return unitary

class HybridFunction(Function):
    """ Hybrid quantum - classical function definition """

    @staticmethod
    def forward(ctx, params, x, qubits, shift):
        """ Forward pass computation """
       # Pauliz measurement
        ctx.qubits =qubits
        cir1 = Circuit()
        cir1.z(0)
        for i in range(1, qubits):
            cir1.i(i)
        M = torch.tensor(cir1.to_unitary()).type(dtype=torch.complex64)
        ctx.M = M
        ctx.shift = shift
        w = params

        # cir2 = Circuit()
        # for which_q in range(0, qubits):
        #     cir2.ry(which_q, w[0+6*which_q])
        #     cir2.rz(which_q, w[1+6*which_q])
        #     cir2.ry(which_q, w[2+6*which_q])
        #     if which_q < (qubits-1):
        #         cir2.cnot(which_q, which_q + 1)
        #     else:
        #         cir2.cnot(which_q, 0)
        #     cir2.ry(which_q, w[3+6*which_q])
        #     cir2.rz(which_q, w[4+6*which_q])
        #     cir2.ry(which_q, w[5+6*which_q])
        #
        # unitary = torch.tensor(cir2.to_unitary(), requires_grad = True).type(dtype=torch.complex64)
        cir2 = qcircuit(w, ctx.qubits)
        unitary = cir2.run()

        if x.shape[0] == 1:
            out = unitary @ x.T
            res = (out.conj().T @ M @ out).real
        else:
            out = unitary @ x.T
            res = (out.conj().T @ M @ out).diag().real
            res = res.reshape(-1,1)

        ctx.save_for_backward(params, x)

        return res

    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass computation """
        params, x = ctx.saved_tensors
        qubits = ctx.qubits
        M = ctx.M
        gradients = []
        for i in range(len(params)):
            wl = copy.deepcopy(params)
            wr = copy.deepcopy(params)
            wl.requires_grad = False
            wr.requires_grad = False
            wl[i] += ctx.shift
            wr[i] -= ctx.shift

            # cir2 = Circuit()
            # for which_q in range(0, qubits):
            #     cir2.ry(which_q, wl[0+6*which_q])
            #     cir2.rz(which_q, wl[1+6*which_q])
            #     cir2.ry(which_q, wl[2+6*which_q])
            #     if which_q < (qubits - 1):
            #         cir2.cnot(which_q, which_q + 1)
            #     else:
            #         cir2.cnot(which_q, 0)
            #     cir2.ry(which_q, wl[3+6*which_q])
            #     cir2.rz(which_q, wl[4+6*which_q])
            #     cir2.ry(which_q, wl[5+6*which_q])
            # unitary = torch.tensor(cir2.to_unitary(), requires_grad=True).type(dtype=torch.complex64)

            cir2 = qcircuit(wl, ctx.qubits)
            unitary = cir2.run()

            if x.shape[0] == 1:
                out = unitary @ x.T
                resl = (out.conj().T @ M @ out).real
            else:
                out = unitary @ x.T
                resl = (out.conj().T @ M @ out).diag().real
                resl = resl.reshape(-1, 1)
                # resl = resl.mean()

            # cir2 = Circuit()
            # for which_q in range(0, qubits):
            #     cir2.ry(which_q, wr[0+6*which_q])
            #     cir2.rz(which_q, wr[1+6*which_q])
            #     cir2.ry(which_q, wr[2+6*which_q])
            #     if which_q < (qubits - 1):
            #         cir2.cnot(which_q, which_q + 1)
            #     else:
            #         cir2.cnot(which_q, 0)
            #     cir2.ry(which_q, wr[3+6*which_q])
            #     cir2.rz(which_q, wr[4+6*which_q])
            #     cir2.ry(which_q, wr[5+6*which_q])
            # unitary = torch.tensor(cir2.to_unitary(), requires_grad=True).type(dtype=torch.complex64)
            
            cir2 = qcircuit(wr, ctx.qubits)
            unitary = cir2.run()

            if x.shape[0] == 1:
                out = unitary @ x.T
                resr = (out.conj().T @ M @ out).real
            else:
                out = unitary @ x.T
                resr = (out.conj().T @ M @ out).diag().real
                resr = resr.reshape(-1, 1)
                # resr = resr.mean()

            gradient = resl - resr
            gradients.append(gradient)
        if len(gradients[0].shape) == 0:
            grad = torch.tensor(gradients).float()
            grad = grad * grad_output
        else:
            grad = torch.tensor([item.numpy() for item in gradients]).float()
            grad = grad.squeeze(2)
            grad = grad.T * grad_output
            grad = grad.mean(0)

        return grad, None, None, None

def foo(x1):
    # y = 2*math.sin(2*x1[0]+1.9) + 3*math.cos(2*x1[1]+1.9) + x1[2] + x1[3]**2
    # y = x1[0]+x1[1]+x1[2]+x1[3]
    y = 2*math.sin(2*(x1[0]+x1[1]+x1[2]+x1[3])+1.9)
    return y


if __name__ == "__main__":

    num_examples = 256
    num_inputs = 4
    num_outputs = 1
    features = torch.empty(num_examples, num_inputs)
    labels = torch.empty(num_examples, num_outputs)

    for i in range(num_examples):
        features[i] = torch.rand(num_inputs) * 2 * math.pi
    features = features + 0j
    features = nn.functional.normalize(features)
    for i in range(num_examples):
        labels[i] = foo(features[i]) + 1e-3 * random.random()

    def data_iter(batch_size, features, labels):
        #输入batch_size，输入训练集地数据features+标签labels
        num_examples = len(features)
        indices = list(range(num_examples))
        random.shuffle(indices) #把indices列表顺序随机打乱
        for i in range(0,num_examples,batch_size):
            #每次取batch_size个训练样本,j代表索引
            j = torch.LongTensor( indices[i:min(i+batch_size,num_examples)] )
            #print(features.index_select(0,j), labels.index_select(0,j))
            yield features.index_select(0,j), labels.index_select(0,j)
            #把张量沿着0维，只保留取出索引号对应的元素

    net1 = CirModel(2)
    loss = nn.MSELoss()
    optimizer = optim.Adam(net1.parameters(), lr=0.01) #lr为学习率
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50,100], gamma=0.1)

    num_epochs = 600
    batch_size = 10

    #记录loss随着epoch的变化，用于后续绘图
    epoch_lst = [i+1 for i in range(num_epochs)]
    loss_lst = []

    for epoch in range(1, num_epochs + 1):
        t1 = time.time()
        for x, y in data_iter(batch_size, features, labels):
            output = net1(x)
            l = loss(output, y)

            # print(l)
            # print(l.requires_grad)
            # 梯度清0
            optimizer.zero_grad()
            l.backward()
            # print('output.grad: ',output.requires_grad)
            '''
            parameters：希望实施梯度裁剪的可迭代网络参数
            max_norm：该组网络参数梯度的范数上限
            norm_type：范数类型(一般默认为L2 范数, 即范数类型=2) 
            torch.nn.utils.clipgrad_norm() 的使用应该在loss.backward() 之后，optimizer.step()之前.
            '''
            # nn.utils.clip_grad_norm_(net1.circuit.weight,max_norm=1,norm_type=2)
            # print('loss: ',l.item())
            # print("weights_grad2:",net1.circuit.weight.grad,'  weight is leaf?:',net1.circuit.weight.is_leaf)
            # grad = net1.circuit.weight.grad
            # net1.circuit.weight.grad \
            #     = torch.where(torch.isnan(grad),torch.full_like(grad, 0),grad)
            optimizer.step()

        lr_scheduler.step()  # 进行学习率的更新
        loss_lst.append(l.item())
        t2 = time.time()
        print("epoch:%d, loss:%f" % (epoch, l.item()),';current lr:', optimizer.state_dict()["param_groups"][0]["lr"],'   耗时：', t2 - t1)

    # print('开始绘图：')
    # plt.cla()
    # plt.subplot(121)
    # xx = list(features[:num_examples, 0])
    #
    # # yy = [float(each) for each in net1( features[:num_examples,:] ).squeeze() ]
    # yy = []
    # for i in range(num_examples):
    #     yy.append(float(net1(features[i:i + 1, :]).squeeze()))
    # # print(yy)
    # xx = [float(xi) for xi in xx]
    # yy_t = [foo(xi) for xi in xx]
    # plt.plot(xx, yy, 'm^', linewidth=1, markersize=2)
    # plt.plot(xx, yy_t, 'g^', linewidth=1, markersize=0.5)
    #
    # plt.subplot(122)
    # plt.plot(epoch_lst, loss_lst, 'r^--', linewidth=1, markersize=1.5)
    # plt.show()