import warnings
#import numpy as np
import flwr as fl
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

#from matplotlib import pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
#from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import DataLoader
#from torchvision.utils import make_grid
from tqdm import tqdm

from utils import *
from dataloader import DTIDataset
# from models import DeepDTA
from network import Conv1d


warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# transform = transforms.Compose([
#     transforms.ToTensor(),  # 把数据转换为张量（Tensor）
#     transforms.Normalize(  # 标准化，即使数据服从期望值为 0，标准差为 1 的正态分布
#         mean=[0.5, ],  # 期望
#         std=[0.5, ]  # 标准差
#     )
# ])
# # 训练集导入
# data_train = datasets.MNIST(root='data/', transform=transform, train=True, download=True)
# # 数据集导入
# data_test = datasets.MNIST(root='data/', transform=transform, train=False)

root_dir = r'./data/'
train_dataset_name = 'training_dataset.csv'
test_dataset_name = 'test_dataset.csv'
val_dataset_name = 'validation_dataset.csv'
TRAIN_PATH = './logs/DTA_result/train/epoch{} train_loss_min_{}_dict_dta.pth'
TEST_PATH = './logs/DTA_result/test/epoch{}_dict_dta.pth'
BEST_RESULT_PATH = './logs/DTA_result/best_result_dict_dta.pth'

train_dataset = DTIDataset(root_dir=root_dir, csv_name=train_dataset_name)
train_dataloader = DataLoader(train_dataset, batch_size=32,shuffle=True,drop_last=True)

test_dataset = DTIDataset(root_dir=root_dir, csv_name=test_dataset_name)
test_dataloader = DataLoader(test_dataset, batch_size=32,shuffle=True,drop_last=True)

val_dataset = DTIDataset(root_dir=root_dir, csv_name=val_dataset_name)
val_dataloader = DataLoader(val_dataset, batch_size=32,shuffle=True,drop_last=True)


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 64, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(1024, 10)
#         #self.fc2 = nn.Linear(50, 10)
 
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 1024)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         #x = self.fc2(x)
#         return F.log_softmax(x, dim=1)
class Net(nn.Module):
    """DeepDTA model architecture, Y-shaped net that does 1d convolution on 
    both the ligand and the protein representation and then concatenates the
    result into a final predictor of binding affinity"""

    def __init__(self, pro_vocab_size, lig_vocab_size, channel, protein_kernel_size, ligand_kernel_size):
        super(Net, self).__init__()
        self.ligand_conv = Conv1d(lig_vocab_size, channel, ligand_kernel_size)
        self.protein_conv = Conv1d(pro_vocab_size, channel, protein_kernel_size)
        self.fc1 = nn.Linear(channel*6, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, protein, ligand):
        x1 = self.ligand_conv(ligand)
        x2 = self.protein_conv(protein)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x
    

net = Net(512,128,32,12,4).to(DEVICE)
#trainloader, testloader = load_data()
# 训练集装载
# dataloader_train = DataLoader(dataset=data_train, batch_size=64, shuffle=True)
# 数据集装载
# dataloader_test = DataLoader(dataset=data_test, batch_size=64, shuffle=True)

trainloader = train_dataloader
testloader = test_dataloader

def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = nn.MSELoss() # torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=0.001) #torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for data in tqdm(trainloader):
            ligand, protein, labels = data
            optimizer.zero_grad()
            criterion(net(ligand.to(DEVICE), protein.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, val_loader):
    """Validate the model on the test set."""
    # criterion = torch.nn.CrossEntropyLoss()
    pred_list = []
    label_list = []
    rmse_list = []
    # correct, loss = 0, 0.0
    with torch.no_grad():
        for data in tqdm(val_loader):
            ligand, protein, labels = data
            outputs = net(ligand.to(DEVICE), protein.to(DEVICE))
            labels = labels.to(DEVICE)
            rmse_list.append(get_rmse(labels, outputs))
            pred_list.append(outputs)
            label_list.append(labels)
            # loss += criterion(outputs, labels).item()
            # correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    # accuracy = correct / len(testloader.dataset)
    rmse = sum(rmse_list) / len(rmse_list)
    pred = np.concatenate(pred_list).reshape(-1)
    label = np.concatenate(label_list).reshape(-1)
    accuracy = np.corrcoef(pred, label)[0, 1]
    return rmse, accuracy

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):#config？？
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=300)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


    
fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=FlowerClient(),
)

    
