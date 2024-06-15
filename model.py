# -*- coding: utf-8 -*-

"""
    @File    : model.py
    @Project : wikipediaMining
"""
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch_geometric.nn import GATConv
import torch

bert_path = "/root/data/jupyter/utils/nlp/model_params/bert-base-cased/"
class BERTGCNModel(nn.Module):
    def __init__(self, num_classes, gcn_hidden_dim):
        super(BERTGCNModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.gcn_hidden_dim = gcn_hidden_dim
        self.gcn_hidden_dim = gcn_hidden_dim
        self.gcn1 = GATConv(self.bert.config.hidden_size, 512, heads=1)
        self.gcn2 = GATConv(512, gcn_hidden_dim, heads=1)
        self.gcn3 = GATConv(self.bert.config.hidden_size, 512, heads=1)
        self.gcn4 = GATConv(512, gcn_hidden_dim, heads=1)
        self.fc = nn.Linear(gcn_hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, custom_mask, graph, sub_graph):
        edge_index = graph.edge_index
        sub_edge_index = sub_graph.edge_index
        # sub_edge_index = custom_index(custom_mask, edge_index)

        # Get BERT context embeddings
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = bert_outputs.last_hidden_state
        # Get BERT core embedding
        core_bert_output = self.bert(input_ids=input_ids, attention_mask=custom_mask)
        core_embeddings = core_bert_output.last_hidden_state

        # Graph Conv
        graph_feats = embeddings.view(input_ids.size()[0] * 86, 768)
        core_graph_feats = core_embeddings.view(input_ids.size()[0] * 86, 768)
        # Pass embeddings through GCN layers
        x = self.gcn1(graph_feats, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.gcn2(x, edge_index)
        x = torch.relu(x)
        # x = self.dropout(x)

        x1 = self.gcn3(core_graph_feats, sub_edge_index)
        x1 = torch.relu(x1)
        # x1 = self.dropout(x1)
        x1 = self.gcn4(x1, sub_edge_index)
        # print(x.size())
        x1 = torch.relu(x1)
        # x1 = self.dropout(x1)

        # # Pass through fully connected layer
        x = x.view(input_ids.size()[0], 86, self.gcn_hidden_dim)
        x1 = x1.view(input_ids.size()[0], 86, self.gcn_hidden_dim)

        max_pool_dim1, max_pool_indices_dim1 = torch.max(x, dim=1, keepdim=True)
        max_pool_dim1 = max_pool_dim1.squeeze(1)

        core_max_pool_dim1, core_max_pool_indices_dim1 = torch.max(x1, dim=1, keepdim=True)
        core_max_pool_dim1 = core_max_pool_dim1.squeeze(1)

        fc_x = torch.cat([max_pool_dim1, core_max_pool_dim1], dim=1)
        x = self.fc(fc_x)
        return x, max_pool_dim1