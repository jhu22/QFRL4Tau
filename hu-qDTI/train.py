from scipy.stats import pearsonr
import os
import sys
sys.path.append('..')
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import json


from utils import *
from dataloader import DTIDataset
from models import DeepDTA,QuDeepDTA

# pccs = pearsonr(x, y)
np.seterr(divide='ignore',invalid='ignore')

def deepdta_val(model,dataloader):
    model.eval()
    
    pred_list = []
    label_list = []
    rmse_list = []
    for data in dataloader:
        ligand, protein, label =  data
        exp = torch.as_tensor(label, dtype=torch.float).unsqueeze(-1).detach().numpy()
        pred = model(ligand, protein).detach().numpy()
        rmse_list.append(get_rmse(exp, pred))
        pred_list.append(pred)
        label_list.append(exp)
    rmse = sum(rmse_list) / len(rmse_list)
    # rmse = get_rmse(label_list, pred_list)
    
    # print('pred_list:', pred_list[0].shape)
    pred = np.concatenate(pred_list).reshape(-1)
    label = np.concatenate(label_list).reshape(-1)
    # print('pred:',pred.shape)
    # print('label', label.shape)
    # print(0 in pred)
    # print(0 in label)
    np.seterr(divide='ignore',invalid='ignore')
    pccs = np.corrcoef(pred, label)[0, 1]
    # pccs = cal_pccs(label_list, pred_list, len(label_list))
    # pccs = np.column_stack(label_list, pred_list)
    # pccs = sum(pccs_list) / len(pccs_list)
    
    return rmse, pccs


if __name__ == '__main__':
    # 处理np计算过程中遇到0除以0的情况
    np.seterr(divide='ignore',invalid='ignore')

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

    model = QuDeepDTA(512,128,32,12,4)
    criterion=nn.MSELoss()
    optimizer=optim.Adam(model.parameters(),lr=0.001)
    epochs = 600

    loss_list = []
    # train_loss_list = []
    # test_rmse_list = []
    # test_pr_list = []

    train_min_loss = 1000
    test_min_rmse = 1000
    test_max_pr = -1

    writer = SummaryWriter("./logs/")

    for epoch in range(epochs):
        model.train()
        for data in train_dataloader:
            ligand, protein, label = data
            pred = model(ligand, protein)
            optimizer.zero_grad()
            loss = criterion(pred.reshape(-1), label.reshape(-1))
            loss_list.append(loss.detach().numpy().tolist())
            loss.backward()
            optimizer.step()
        train_loss = sum(loss_list) / len(loss_list)
        
        if train_loss < train_min_loss:
            train_min_loss = train_loss
            torch.save(model.state_dict(), TRAIN_PATH.format(str(epoch),str(train_loss)[7:13]))

        # start validatin
        test_rmse, test_pr = deepdta_val(model, test_dataloader)

        # log
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("test_rmse", test_rmse, epoch)
        writer.add_scalar("test_pr", test_pr, epoch)
        # train_loss_list.append(train_loss)
        # test_rmse_list.append(test_rmse)
        # test_pr_list.append(test_pr)
        
        if test_rmse < test_min_rmse or test_pr > test_max_pr:
            if test_rmse < test_min_rmse and test_pr > test_max_pr:
                test_min_rmse = test_rmse
                test_max_pr = test_pr
                torch.save(model.state_dict(), BEST_RESULT_PATH)
                print('best result-epochs:', epoch + 1, 'train-loss:', '%.4f' % train_loss)
                print('valid-rmse:', '% .4f'%test_rmse, 'valid-pr:','% .4f'%test_pr)
                continue
            elif test_rmse < test_min_rmse :
                test_min_rmse = test_rmse
            else:
                test_max_pr = test_pr
            torch.save(model.state_dict(), TEST_PATH.format(str(epoch)))
        

        print('epochs:', epoch + 1, 'train-loss:', '%.4f' % train_loss)
        print('valid-rmse:', '% .4f'%test_rmse, 'valid-pr:','% .4f'%test_pr)

    model.load_state_dict(torch.load(BEST_RESULT_PATH)) 
    val_rmse, val_pr = deepdta_val(model, val_dataloader)
    print('valid-rmse:', '% .4f'%val_rmse, 'valid-pr:','% .4f'%val_pr)

    # save_path = "./data/DTA_result/"
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # dict = {"train_loss": train_loss_list,"rmse":test_rmse_list, "pr":test_pr_list}
    # with open(save_path + "DTA_result.json", "w") as f:
    #     json.dump(dict, f)
