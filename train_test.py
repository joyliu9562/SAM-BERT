# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import seed_everything, contrastive_loss
from data_loader import MyOwnDataset_Test, MyOwnDataset_Train
from torch_geometric.loader import DataLoader
from model import BERTGCNModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

device = 'cuda'
seed_everything(88)
with open('./data/sentence_idx.pkl', 'rb') as file:
    # load file dict\
    file_dic = pickle.load(file)
train_set = MyOwnDataset_Train(root='./data/unbalance_aug/', file_dic=file_dic)
test_set = MyOwnDataset_Test(root='./data/unbalance_aug/', file_dic=file_dic)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)
model = BERTGCNModel(23, 256).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)

acc = []
statics = []
gat = []
for epoch in range(50):
    ce_loss = 0
    contra_loss = 0
    model.train()
    print("Epoch : #", epoch + 1)
    for idx, row in enumerate(train_loader):
        # the order of the attention_mask and costm_mask is different in dataloader 
        graph, sub_graph, input_ids, attention_mask, costm_mask = row
        input_ids = input_ids.squeeze(1).to(device)
        attention_mask = attention_mask.squeeze(1).to(device)
        costm_mask = costm_mask.squeeze(1).to(device)
        graph = graph.to(device)
        sub_graph = sub_graph.to(device)
        y = graph.y.to(device)
        output, gcn_feats = model(input_ids, attention_mask, costm_mask, graph, sub_graph)
        # contrastive loss
        contrast_loss = contrastive_loss(0.1, gcn_feats.cpu().detach().numpy(), y)
        ce_loss = criterion(output, y)
        loss = 0.9 * contrast_loss + 0.1 * ce_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        contra_loss += contrast_loss / len(y)
        ce_loss += ce_loss.item() / len(y)
    print("Contra Loss: ", contra_loss, "CE Loss: ", ce_loss.item())
    model.eval()
    val_loss = 0
    score = None
    total_pred = []
    total_labels = []
    total_source = []
    for idx, row in enumerate(test_loader):       
        # the order of the attention_mask and costm_mask is different in dataloader 
        graph, sub_graph, input_ids, attention_mask, costm_mask = row
        input_ids = input_ids.squeeze(1).to(device)
        attention_mask = attention_mask.squeeze(1).to(device)
        costm_mask = costm_mask.squeeze(1).to(device)
        graph = graph.to(device)
        sub_graph = sub_graph.to(device)
        y = graph.y.to(device)

        output, gcn_feats = model(input_ids, attention_mask, costm_mask, graph, sub_graph)
        _, predicted = torch.max(output.data, 1)
        total_pred += list(predicted.cpu().numpy().reshape(-1))
        total_labels += list(y.cpu().numpy().reshape(-1))
        # val_loss += loss.item()
    score = accuracy_score(total_labels, total_pred)
    recision, recall, f1, _ = precision_recall_fscore_support(total_labels, total_pred, average='weighted')
    print(score, recision, recall, f1)
    acc.append((score, recision, recall, f1))
