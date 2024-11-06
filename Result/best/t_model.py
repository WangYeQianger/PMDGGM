# -*- coding:utf-8 -*-
# 作者: 沐晨汐__
# 日期: 2024-02-21
# 声明: Welcome my coding environments!
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATConv, SAGPooling, SAGEConv, TopKPooling, GINConv, TAGConv, \
    CuGraphRGCNConv, PointNetConv, HEATConv, SuperGATConv, FiLMConv, PointTransformerConv, FastRGCNConv, \
    TransformerConv, GravNetConv, SignedConv, RGATConv
import torch_geometric.transforms as T
from torch_geometric.utils import degree
from torch_sparse import SparseTensor


class mySAGEConv_(SAGEConv):
    def __init__(self, in_channels, out_channels, miRNA_disease_dim=512):
        super(mySAGEConv_, self).__init__(in_channels, out_channels)
        self.miRNA_linear_same = nn.Linear(in_channels, miRNA_disease_dim)
        self.miRNA_linear_diff = nn.Linear(in_channels, miRNA_disease_dim)
        self.disease_linear_same = nn.Linear(in_channels, miRNA_disease_dim)
        self.disease_linear_diff = nn.Linear(in_channels, miRNA_disease_dim)
        self.out_channels = out_channels

        # 双线性池化融合层
        self.bilinear_pool = nn.Bilinear(miRNA_disease_dim, miRNA_disease_dim, out_channels)

        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 计算相同类型邻居节点的聚合特征
        miRNA_mask_same = (row < 812) & (col < 812)
        disease_mask_same = (row >= 812) & (col >= 812)

        miRNA_feats_same = torch.zeros_like(x)
        disease_feats_same = torch.zeros_like(x)

        miRNA_feats_same.index_add_(0, row[miRNA_mask_same],
                                    x[col[miRNA_mask_same]] * norm[miRNA_mask_same].unsqueeze(-1))
        disease_feats_same.index_add_(0, row[disease_mask_same],
                                      x[col[disease_mask_same]] * norm[disease_mask_same].unsqueeze(-1))

        # 计算不同类型邻居节点的聚合特征
        miRNA_mask_diff = (row < 812) & (col >= 812)
        disease_mask_diff = (row >= 812) & (col < 812)

        miRNA_feats_diff = torch.zeros_like(x)
        disease_feats_diff = torch.zeros_like(x)

        miRNA_feats_diff.index_add_(0, row[miRNA_mask_diff],
                                    x[col[miRNA_mask_diff]] * norm[miRNA_mask_diff].unsqueeze(-1))
        disease_feats_diff.index_add_(0, row[disease_mask_diff],
                                      x[col[disease_mask_diff]] * norm[disease_mask_diff].unsqueeze(-1))

        # 将miRNA和疾病节点的特征通过线性层
        miRNA_feats_same = self.miRNA_linear_same(miRNA_feats_same)
        disease_feats_same = self.disease_linear_same(disease_feats_same)

        miRNA_feats_diff = self.miRNA_linear_diff(miRNA_feats_diff)
        disease_feats_diff = self.disease_linear_diff(disease_feats_diff)

        # 双线性池化融合
        bilinear_feats = self.bilinear_pool(miRNA_feats_diff, disease_feats_diff)

        # 将 miRNA_feats_diff 和 disease_feats_diff 的尺寸调整为 out_channels
        miRNA_feats_diff = miRNA_feats_diff[:, :self.out_channels]
        disease_feats_diff = disease_feats_diff[:, :self.out_channels]

        # 裁剪 miRNA_feats_same 和 disease_feats_same 的尺寸为 out_channels
        miRNA_feats_same = miRNA_feats_same[:, :self.out_channels]
        disease_feats_same = disease_feats_same[:, :self.out_channels]

        # 使用门控机制调整融合特征
        gate = self.gate(bilinear_feats)
        fused_feats = gate * F.leaky_relu(bilinear_feats) + (1 - gate) * (miRNA_feats_diff + disease_feats_diff)

        # 最终特征组合
        combined_feats = fused_feats + miRNA_feats_same + disease_feats_same

        return combined_feats + x[:, :self.out_channels]


class mySAGEConv_Test(nn.Module):
    def __init__(self, num_features):
        super(mySAGEConv_Test, self).__init__()
        self.attention = GATConv(num_features, 1024 // 8, heads=8, dropout=0.5)
        self.conv1 = mySAGEConv_(1024, 512)
        self.conv2 = mySAGEConv_(512, 256)
        self.conv3 = mySAGEConv_(256, 128)
        self.lin = nn.Linear(128, 1)
        self.lin_edge = nn.Linear(256, 1)

    def forward(self, x, edge_index, edge_samples):
        x = F.leaky_relu(self.attention(x, edge_index))
        attention_out = x

        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.leaky_relu(self.conv3(x, edge_index))

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


class TestConv(SAGEConv):
    def __init__(self, in_channels, out_channels):
        super(TestConv, self).__init__(in_channels, out_channels)
        self.miRNA_linear_same = nn.Linear(in_channels, out_channels)
        self.miRNA_linear_diff = nn.Linear(in_channels, out_channels)
        self.disease_linear_same = nn.Linear(in_channels, out_channels)
        self.disease_linear_diff = nn.Linear(in_channels, out_channels)
        self.out_channels = out_channels
        self.bilinear_pool = nn.Bilinear(out_channels, out_channels, out_channels)
        self.gate = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 1. 处理同质特征聚合(只考虑有边相连的)
        # miRNA-miRNA
        miRNA_mask_same = (row < 812) & (col < 812)
        miRNA_feats_same = torch.zeros_like(x)
        miRNA_connected_pairs = edge_index[:, miRNA_mask_same]
        if len(miRNA_connected_pairs) > 0:
            miRNA_feats_same.index_add_(0,
                                        miRNA_connected_pairs[0],
                                        x[miRNA_connected_pairs[1]] * norm[miRNA_mask_same].unsqueeze(-1)
                                        )

        # disease-disease
        disease_mask_same = (row >= 812) & (col >= 812)
        disease_feats_same = torch.zeros_like(x)
        disease_connected_pairs = edge_index[:, disease_mask_same]
        if len(disease_connected_pairs) > 0:
            disease_feats_same.index_add_(0,
                                          disease_connected_pairs[0],
                                          x[disease_connected_pairs[1]] * norm[disease_mask_same].unsqueeze(-1)
                                          )

        # 2. 处理异构特征聚合
        miRNA_mask_diff = (row < 812) & (col >= 812)  # miRNA->disease
        disease_mask_diff = (row >= 812) & (col < 812)  # disease->miRNA

        # miRNA节点聚合所有相连疾病节点的特征
        miRNA_feats_diff = torch.zeros_like(x)
        miRNA_disease_pairs = edge_index[:, miRNA_mask_diff]
        if len(miRNA_disease_pairs) > 0:
            miRNA_feats_diff.index_add_(0,
                                        miRNA_disease_pairs[0],
                                        x[miRNA_disease_pairs[1]] * norm[miRNA_mask_diff].unsqueeze(-1)
                                        )

        # 疾病节点聚合所有相连miRNA节点的特征
        disease_feats_diff = torch.zeros_like(x)
        disease_miRNA_pairs = edge_index[:, disease_mask_diff]
        if len(disease_miRNA_pairs) > 0:
            disease_feats_diff.index_add_(0,
                                          disease_miRNA_pairs[0],
                                          x[disease_miRNA_pairs[1]] * norm[disease_mask_diff].unsqueeze(-1)
                                          )

        # 3. 线性变换
        miRNA_feats_same = self.miRNA_linear_same(miRNA_feats_same)
        disease_feats_same = self.disease_linear_same(disease_feats_same)
        miRNA_feats_diff = self.miRNA_linear_diff(miRNA_feats_diff)
        disease_feats_diff = self.disease_linear_diff(disease_feats_diff)

        # 4. 双线性池化(针对每个miRNA和其相连的disease)
        bilinear_feats = torch.zeros(x.size(0), self.out_channels).to(x.device)

        # 处理每个miRNA节点
        unique_miRNAs = torch.unique(miRNA_disease_pairs[0])
        for miRNA_idx in unique_miRNAs:
            # 找到与当前miRNA相连的所有disease
            curr_miRNA_edges = miRNA_disease_pairs[0] == miRNA_idx
            connected_diseases = miRNA_disease_pairs[1, curr_miRNA_edges]

            # 获取当前miRNA节点的异构特征
            curr_miRNA_hete = miRNA_feats_diff[miRNA_idx]

            # 收集所有相连disease的pooling结果
            pooled_results = []
            for disease_idx in connected_diseases:
                # 获取当前disease节点的异构特征
                curr_disease_hete = disease_feats_diff[disease_idx]

                # 对当前miRNA-disease对进行双线性池化
                curr_pool = self.bilinear_pool(
                    curr_miRNA_hete.unsqueeze(0),
                    curr_disease_hete.unsqueeze(0)
                ).squeeze(0)  # 确保移除批次维度
                pooled_results.append(curr_pool)

            # 平均所有pooling结果作为该miRNA节点的双线性特征
            if pooled_results:
                miRNA_bilinear = torch.stack(pooled_results).mean(dim=0)
                bilinear_feats[miRNA_idx] = miRNA_bilinear

        # 处理每个disease节点(类似地)
        unique_diseases = torch.unique(disease_miRNA_pairs[0])
        for disease_idx in unique_diseases:
            curr_disease_edges = disease_miRNA_pairs[0] == disease_idx
            connected_miRNAs = disease_miRNA_pairs[1, curr_disease_edges]

            curr_disease_hete = disease_feats_diff[disease_idx]

            pooled_results = []
            for miRNA_idx in connected_miRNAs:
                curr_miRNA_hete = miRNA_feats_diff[miRNA_idx]
                curr_pool = self.bilinear_pool(
                    curr_disease_hete.unsqueeze(0),
                    curr_miRNA_hete.unsqueeze(0)
                ).squeeze(0)  # 确保移除批次维度
                pooled_results.append(curr_pool)

            if pooled_results:
                disease_bilinear = torch.stack(pooled_results).mean(dim=0)
                bilinear_feats[disease_idx] = disease_bilinear

        # 5. 使用门控机制
        gate = self.gate(bilinear_feats)
        fused_feats = gate * F.leaky_relu(bilinear_feats) + \
                      (1 - gate) * (miRNA_feats_diff + disease_feats_diff)

        # 6. 最终特征组合
        combined_feats = fused_feats + miRNA_feats_same + disease_feats_same

        return combined_feats


class TestConv1(SAGEConv):
    def __init__(self, in_channels, out_channels):
        super(TestConv1, self).__init__(in_channels, out_channels)
        self.miRNA_linear_same = nn.Linear(in_channels, out_channels)
        self.miRNA_linear_diff = nn.Linear(in_channels, out_channels)
        self.disease_linear_same = nn.Linear(in_channels, out_channels)
        self.disease_linear_diff = nn.Linear(in_channels, out_channels)
        self.out_channels = out_channels
        self.bilinear_pool = nn.Bilinear(out_channels, out_channels, out_channels)
        self.gate = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 1. 同质特征聚合
        miRNA_mask_same = (row < 812) & (col < 812)
        disease_mask_same = (row >= 812) & (col >= 812)

        miRNA_feats_same = torch.zeros_like(x)
        disease_feats_same = torch.zeros_like(x)

        # 使用masked_select进行批量处理
        if miRNA_mask_same.any():
            miRNA_feats_same.index_add_(0,
                                        row[miRNA_mask_same],
                                        x[col[miRNA_mask_same]] * norm[miRNA_mask_same].unsqueeze(-1))

        if disease_mask_same.any():
            disease_feats_same.index_add_(0,
                                          row[disease_mask_same],
                                          x[col[disease_mask_same]] * norm[disease_mask_same].unsqueeze(-1))

        # 2. 异构特征聚合
        miRNA_mask_diff = (row < 812) & (col >= 812)
        disease_mask_diff = (row >= 812) & (col < 812)

        miRNA_feats_diff = torch.zeros_like(x)
        disease_feats_diff = torch.zeros_like(x)

        if miRNA_mask_diff.any():
            miRNA_feats_diff.index_add_(0,
                                        row[miRNA_mask_diff],
                                        x[col[miRNA_mask_diff]] * norm[miRNA_mask_diff].unsqueeze(-1))

        if disease_mask_diff.any():
            disease_feats_diff.index_add_(0,
                                          row[disease_mask_diff],
                                          x[col[disease_mask_diff]] * norm[disease_mask_diff].unsqueeze(-1))

        # 3. 线性变换
        miRNA_feats_same = self.miRNA_linear_same(miRNA_feats_same)
        disease_feats_same = self.disease_linear_same(disease_feats_same)
        miRNA_feats_diff = self.miRNA_linear_diff(miRNA_feats_diff)
        disease_feats_diff = self.disease_linear_diff(disease_feats_diff)

        # 4. 批量双线性池化
        bilinear_feats = torch.zeros(x.size(0), self.out_channels).to(x.device)

        # 获取miRNA-disease边
        miRNA_disease_edges = edge_index[:, miRNA_mask_diff]
        if len(miRNA_disease_edges) > 0:
            # 批量处理所有miRNA节点的双线性池化
            miRNA_features = miRNA_feats_diff[miRNA_disease_edges[0]]  # [E, out_channels]
            disease_features = disease_feats_diff[miRNA_disease_edges[1]]  # [E, out_channels]

            # 批量双线性池化
            edge_bilinear = self.bilinear_pool(miRNA_features, disease_features)  # [E, out_channels]

            # 对每个miRNA节点，平均其所有相连disease的池化结果
            bilinear_feats.index_add_(0, miRNA_disease_edges[0], edge_bilinear)
            edge_counts = torch.zeros(x.size(0), device=x.device)
            edge_counts.index_add_(0, miRNA_disease_edges[0],
                                   torch.ones_like(miRNA_disease_edges[0], dtype=torch.float))
            edge_counts = torch.clamp(edge_counts, min=1)  # 避免除0
            bilinear_feats[:812] = bilinear_feats[:812] / edge_counts[:812].unsqueeze(-1)

            # 同样处理disease节点
            bilinear_feats.index_add_(0, miRNA_disease_edges[1], edge_bilinear)
            edge_counts = torch.zeros(x.size(0), device=x.device)
            edge_counts.index_add_(0, miRNA_disease_edges[1],
                                   torch.ones_like(miRNA_disease_edges[1], dtype=torch.float))
            edge_counts = torch.clamp(edge_counts, min=1)
            bilinear_feats[812:] = bilinear_feats[812:] / edge_counts[812:].unsqueeze(-1)

        # 5. 门控机制
        gate = self.gate(bilinear_feats)
        fused_feats = gate * F.leaky_relu(bilinear_feats) + \
                      (1 - gate) * (miRNA_feats_diff + disease_feats_diff)

        # 6. 最终特征组合
        combined_feats = fused_feats + miRNA_feats_same + disease_feats_same

        return combined_feats


class TestConv2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TestConv2, self).__init__()
        self.linear_same = nn.Linear(in_channels, out_channels)
        self.linear_diff = nn.Linear(in_channels, out_channels)
        self.bilinear_pool = nn.Bilinear(out_channels, out_channels, out_channels)
        self.gate_weight = nn.Parameter(torch.Tensor(out_channels))
        nn.init.constant_(self.gate_weight, 0.5)

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 同质邻居
        miRNA_mask_same = (row < 812) & (col < 812)
        disease_mask_same = (row >= 812) & (col >= 812)
        mask_same = miRNA_mask_same | disease_mask_same
        row_same = row[mask_same]
        col_same = col[mask_same]
        norm_same = norm[mask_same]

        A_same = SparseTensor(row=row_same, col=col_same, value=norm_same,
                              sparse_sizes=(x.size(0), x.size(0)))
        x_same = A_same @ x  # [N, in_channels]
        x_same_transformed = self.linear_same(x_same)

        # 异质邻居
        miRNA_mask_diff = (row < 812) & (col >= 812)
        disease_mask_diff = (row >= 812) & (col < 812)
        mask_diff = miRNA_mask_diff | disease_mask_diff
        row_diff = row[mask_diff]
        col_diff = col[mask_diff]
        norm_diff = norm[mask_diff]

        A_diff = SparseTensor(row=row_diff, col=col_diff, value=norm_diff,
                              sparse_sizes=(x.size(0), x.size(0)))
        x_diff = A_diff @ x  # [N, in_channels]
        x_diff_transformed = self.linear_diff(x_diff)

        # 双线性池化
        miRNA_disease_edges = edge_index[:, miRNA_mask_diff]
        if miRNA_disease_edges.size(1) > 0:
            miRNA_features = x_diff_transformed[miRNA_disease_edges[0]]
            disease_features = x_diff_transformed[miRNA_disease_edges[1]]
            edge_bilinear = self.bilinear_pool(miRNA_features, disease_features)

            # 聚合边特征到节点
            row_indices = torch.cat([miRNA_disease_edges[0], miRNA_disease_edges[1]])
            edge_indices = torch.arange(edge_bilinear.size(0), device=x.device)
            edge_indices = edge_indices.repeat(2)
            values = torch.ones_like(row_indices, dtype=torch.float)

            A_bilinear = SparseTensor(row=row_indices, col=edge_indices, value=values,
                                      sparse_sizes=(x.size(0), edge_bilinear.size(0)))
            node_bilinear = A_bilinear @ edge_bilinear  # [N, out_channels]

            # 计算关联的边数
            edge_counts = A_bilinear.sum(dim=1).unsqueeze(-1).clamp(min=1)
            bilinear_feats = node_bilinear / edge_counts
        else:
            bilinear_feats = torch.zeros_like(x_diff_transformed)

        # 门控机制
        gate = torch.sigmoid(self.gate_weight)
        fused_feats = gate * F.leaky_relu(bilinear_feats) + (1 - gate) * x_diff_transformed

        # 最终特征组合
        combined_feats = x_same_transformed + fused_feats

        return combined_feats


class TestConv3(nn.Module):
    def __init__(self, in_channels, out_channels, k=3):
        super(TestConv3, self).__init__()
        self.linear_same = nn.Linear(in_channels, out_channels)
        self.linear_diff = nn.Linear(in_channels, out_channels)
        self.bilinear_pool = nn.Bilinear(out_channels, out_channels, out_channels)
        self.gate_weight = nn.Parameter(torch.Tensor(out_channels))
        self.k = k
        nn.init.constant_(self.gate_weight, 0.5)

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 同质邻居
        miRNA_mask_same = (row < 812) & (col < 812)
        disease_mask_same = (row >= 812) & (col >= 812)
        mask_same = miRNA_mask_same | disease_mask_same
        row_same = row[mask_same]
        col_same = col[mask_same]
        norm_same = norm[mask_same]

        A_same = SparseTensor(row=row_same, col=col_same, value=norm_same,
                              sparse_sizes=(x.size(0), x.size(0)))
        x_same = A_same @ x  # [N, in_channels]
        x_same_transformed = self.linear_same(x_same)

        # 异质邻居
        miRNA_mask_diff = (row < 812) & (col >= 812)
        disease_mask_diff = (row >= 812) & (col < 812)
        mask_diff = miRNA_mask_diff | disease_mask_diff
        row_diff = row[mask_diff]
        col_diff = col[mask_diff]
        norm_diff = norm[mask_diff]

        A_diff = SparseTensor(row=row_diff, col=col_diff, value=norm_diff,
                              sparse_sizes=(x.size(0), x.size(0)))
        x_diff = A_diff @ x  # [N, in_channels]
        x_diff_transformed = self.linear_diff(x_diff)

        # 双线性池化,只聚合k个最相关的异质邻居节点特征
        miRNA_indices, disease_indices = torch.where(miRNA_mask_diff)[0], torch.where(disease_mask_diff)[0]

        bilinear_feats = torch.zeros_like(x_diff_transformed)

        for miRNA_idx in miRNA_indices:
            disease_neighbors = disease_indices[edge_index[0, disease_mask_diff] == miRNA_idx]
            if len(disease_neighbors) > 0:
                topk_disease_indices = disease_neighbors[
                    norm_diff[disease_mask_diff][edge_index[0, disease_mask_diff] == miRNA_idx].topk(
                        min(self.k, len(disease_neighbors))).indices]
                miRNA_feature = x_diff_transformed[miRNA_idx].unsqueeze(0)
                disease_features = x_diff_transformed[topk_disease_indices]
                edge_bilinear = self.bilinear_pool(miRNA_feature.expand(len(topk_disease_indices), -1),
                                                   disease_features)
                bilinear_feats[miRNA_idx] = edge_bilinear.mean(dim=0)

        for disease_idx in disease_indices:
            miRNA_neighbors = miRNA_indices[edge_index[1, miRNA_mask_diff] == disease_idx]
            if len(miRNA_neighbors) > 0:
                topk_miRNA_indices = miRNA_neighbors[
                    norm_diff[miRNA_mask_diff][edge_index[1, miRNA_mask_diff] == disease_idx].topk(
                        min(self.k, len(miRNA_neighbors))).indices]
                disease_feature = x_diff_transformed[disease_idx].unsqueeze(0)
                miRNA_features = x_diff_transformed[topk_miRNA_indices]
                edge_bilinear = self.bilinear_pool(disease_feature.expand(len(topk_miRNA_indices), -1), miRNA_features)
                bilinear_feats[disease_idx] = edge_bilinear.mean(dim=0)

        # 门控机制
        gate = torch.sigmoid(self.gate_weight)
        fused_feats = gate * F.leaky_relu(bilinear_feats) + (1 - gate) * x_diff_transformed

        # 最终特征组合
        combined_feats = x_same_transformed + fused_feats

        return combined_feats


class Test(nn.Module):
    def __init__(self, num_features):
        super(Test, self).__init__()
        self.attention = GATConv(num_features, 256 // 32, heads=32, dropout=0.5)
        self.conv1 = TestConv3(256, 128)
        # self.conv2 = TestConv3(256, 128)
        # self.conv3 = TestConv2(512, 128)
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