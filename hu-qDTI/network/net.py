
import torch
import torch.nn as nn
import torch.functional as F

embedding_num_drug = 64      #字典序
embedding_num_target = 25    #字典序
embedding_dim_drug = 16      #2^6
embedding_dim_target = 16    #2^6
hyber_para = 16              #2^4
qubits_cirAorB = 4           #每一边的qubits数
dim_embed = hyber_para

class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.FC1 = nn.Linear(128,32)
        self.FC2 = nn.Linear(32,1)
    def forward(self,x):
        out = F.leaky_relu(self.FC1(x))
        out = F.leaky_relu(self.FC2(out))
        return out
    
class CNNLayerBased(nn.Module):
    def __init__(self, embedding_num_drug, embedding_num_target, embedding_dim_drug=dim_embed,
                  embedding_dim_target=dim_embed, conv1_out_dim = qubits_cirAorB):
        super().__init__()
        # self.data_pre = ClassicalPre()
        self.drugconv1d = nn.Conv1d(embedding_dim_drug, conv1_out_dim, kernel_size = 4, stride = 1, padding = 'same')
        self.targetconv1d = nn.Conv1d(embedding_dim_target, conv1_out_dim, kernel_size = 4, stride = 1, padding = 'same')
        self.linearlayer = Linear()
    def forward(self,drug_input, target_input):   #x是一条记录
        # drug_input, target_input = self.data_pre(x)
        drug_output = self.drugconv1d(drug_input)   #1 4 16

        target_output = self.targetconv1d(target_input)   #1 4 16

        linear_input = torch.cat([drug_output, target_output],dim=1).view(drug_output.shape[0],-1)
        # print('linear',linear_input.shape)
        linear_output = self.linearlayer(linear_input)
        affinity = linear_output
        return affinity

class Conv1d(nn.Module):
    """
    Three 1d convolutional layer with relu activation stacked on top of each other
    with a final global maxpooling layer
    """
    def __init__(self, vocab_size, channel, kernel_size, stride=1, padding=0):
        super(Conv1d, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=128)
        self.conv1 = nn.Conv1d(128, channel, kernel_size, stride, padding)
        self.conv2 = nn.Conv1d(channel, channel*2, kernel_size, stride, padding)
        self.conv3 = nn.Conv1d(channel*2, channel*3, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.globalmaxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = self.embedding(x)
        
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.globalmaxpool(x)
        x = x.squeeze(-1)
        return x
