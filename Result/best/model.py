# -*- coding:utf-8 -*-
# 作者: 王业强__
# 日期: 2024-08-15
# 声明: Welcome my coding environments!

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import degree
from torch_sparse import SparseTensor

class GatedGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=3):
        super(GatedGraphConv, self).__init__()
        self.linear_homo = nn.Linear(in_channels, out_channels)
        self.linear_hete = nn.Linear(in_channels, out_channels)
        self.bilinear_pool = nn.Bilinear(out_channels, out_channels, out_channels)
        self.gate_weight = nn.Parameter(torch.Tensor(out_channels))
        self.k = k
        nn.init.constant_(self.gate_weight, 0.5)

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 聚合同质邻居特征
        miRNA_mask_homo = (row < 812) & (col < 812)
        disease_mask_homo = (row >= 812) & (col >= 812)
        mask_homo = miRNA_mask_homo | disease_mask_homo
        row_homo = row[mask_homo]
        col_homo = col[mask_homo]
        norm_homo = norm[mask_homo]

        A_homo = SparseTensor(row=row_homo, col=col_homo, value=norm_homo,
                              sparse_sizes=(x.size(0), x.size(0)))
        x_homo = A_homo @ x
        x_homo_transformed = self.linear_homo(x_homo)

        # 聚合异质邻居特征
        miRNA_mask_hete = (row < 812) & (col >= 812)
        disease_mask_hete = (row >= 812) & (col < 812)
        mask_hete = miRNA_mask_hete | disease_mask_hete
        row_hete = row[mask_hete]
        col_hete = col[mask_hete]
        norm_hete = norm[mask_hete]

        A_hete = SparseTensor(row=row_hete, col=col_hete, value=norm_hete,
                              sparse_sizes=(x.size(0), x.size(0)))
        x_hete = A_hete @ x
        x_hete_transformed = self.linear_hete(x_hete)

        # 双线性池化,只聚合k个最相关的异质邻居节点特征
        miRNA_indices, disease_indices = torch.where(miRNA_mask_hete)[0], torch.where(disease_mask_hete)[0]

        bilinear_feats = torch.zeros_like(x_hete_transformed)

        for miRNA_idx in miRNA_indices:
            disease_neighbors = disease_indices[edge_index[0, miRNA_mask_hete] == miRNA_idx]
            if len(disease_neighbors) > 0:
                topk_disease_indices = disease_neighbors[
                    norm_hete[miRNA_mask_hete][edge_index[0, miRNA_mask_hete] == miRNA_idx].topk(
                        min(self.k, len(disease_neighbors))).indices]
                miRNA_feature = x_hete_transformed[miRNA_idx].unsqueeze(0)
                disease_features = x_hete_transformed[topk_disease_indices]
                miRNA_bilinear = self.bilinear_pool(miRNA_feature.expand(len(topk_disease_indices), -1),
                                                   disease_features)
                bilinear_feats[miRNA_idx] = miRNA_bilinear.mean(dim=0)

        for disease_idx in disease_indices:
            miRNA_neighbors = miRNA_indices[edge_index[1, disease_mask_hete] == disease_idx]
            if len(miRNA_neighbors) > 0:
                topk_miRNA_indices = miRNA_neighbors[
                    norm_hete[disease_mask_hete][edge_index[1, disease_mask_hete] == disease_idx].topk(
                        min(self.k, len(miRNA_neighbors))).indices]
                disease_feature = x_hete_transformed[disease_idx].unsqueeze(0)
                miRNA_features = x_hete_transformed[topk_miRNA_indices]
                disease_bilinear = self.bilinear_pool(disease_feature.expand(len(topk_miRNA_indices), -1),
                                                   miRNA_features)
                bilinear_feats[disease_idx] = disease_bilinear.mean(dim=0)

        # 门控机制
        gate = torch.sigmoid(self.gate_weight)
        fused_feats = gate * F.leaky_relu(bilinear_feats) + (1 - gate) * x_hete_transformed

        # 最终特征组合
        combined_feats = x_homo_transformed + fused_feats

        return combined_feats


class PMDGGM(nn.Module):
    def __init__(self, num_features):
        super(PMDGGM, self).__init__()
        self.attention = GATConv(num_features, 256 // 32, heads=32, dropout=0.5)
        self.conv1 = GatedGraphConv(256, 128)
        # self.conv2 = GatedGraphConv(256, 128)
        # self.conv3 = GatedGraphConv(512, 128)
        self.lin = nn.Linear(128, 1)
        self.lin_edge = nn.Linear(256, 1)

    def forward(self, x, edge_index, edge_samples):
        x = F.leaky_relu(self.attention(x, edge_index))
        attention_out = x
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.leaky_relu(self.conv1(x, edge_index))

        # x = F.leaky_relu(self.conv2(x, edge_index))
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = F.leaky_relu(self.conv3(x, edge_index))

        embedding = x
        edge_features = self.compute_edge_features(embedding, edge_samples)
        out_edge_logits = self.lin_edge(edge_features).squeeze()

        x = self.lin(embedding).squeeze()
        return x, embedding, edge_features, attention_out, out_edge_logits

    def compute_edge_features(self, node_embeddings, edge_samples):
        embeddings_i = node_embeddings[edge_samples[:, 0]]
        embeddings_j = node_embeddings[edge_samples[:, 1]]
        edge_embeddings = torch.cat([embeddings_i, embeddings_j], dim=1)
        return edge_embeddings