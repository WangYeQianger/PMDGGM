# -*- coding:utf-8 -*-
# 作者: 王业强__
# 日期: 2024-08-15
# 声明: Welcome my coding environments!

import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import networkx as nx
import torch
from sklearn.preprocessing import StandardScaler



def load_data():

    # 加载 M,D,M-D 矩阵
    M = pd.read_csv('./datasets/M.csv', header=None).values
    D = pd.read_csv('./datasets/D.csv', header=None).values
    M_D = pd.read_csv('./datasets/M-D.csv', header=None).values

    # 加载训练用到的正负样本
    all_samples = pd.read_csv('./datasets/samples/samples.csv').values
    labels = pd.read_csv('./datasets/samples/labels.csv').values

    all_samples = torch.tensor(all_samples, dtype=torch.long)
    labels = torch.tensor(labels.squeeze(), dtype=torch.float)
    # print("all_samples.shape: ", all_samples.shape)
    # print("labels.shape: ", labels.shape)

    # 处理矩阵
    M = torch.tensor(M, dtype=torch.float)
    D = torch.tensor(D, dtype=torch.float)
    M_D = torch.tensor(M_D, dtype=torch.float)
    D_M = M_D.T
    combined_matrix = torch.cat((
        torch.cat((M, M_D), dim=1),
        torch.cat((D_M, D), dim=1)
    ), dim=0)

    # 读取特征矩阵
    features = pd.read_excel('./datasets/features/features.xlsx', header=None).values

    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features = torch.tensor(features, dtype=torch.float)
    features = torch.nan_to_num(features, nan=0.0)
    features = (features > 0).float()

    # 构建边索引
    edge_index = (M_D > 0).nonzero(as_tuple=False).t().contiguous()
    edge_index[1, :] += M_D.shape[0]  # 修正边索引
    edge_index_else = edge_index.flip(dims=[0])
    edge_index = torch.cat([edge_index, edge_index_else], dim=1)
    edge_attr = torch.ones(edge_index.shape[1])

    return features, edge_index, edge_attr, all_samples, labels

# 五折交叉验证重新计算边集
def get_MD(edge_index):
    edge_index = edge_index.T
    num_miRNAs = 812
    num_diseases = 438

    M_D = np.zeros((num_miRNAs, num_diseases))
    D_M = np.zeros((num_diseases, num_miRNAs))

    for miRNA_idx, disease_idx in edge_index:
        M_D[miRNA_idx, disease_idx - 812] = 1
        D_M[disease_idx - 812, miRNA_idx] = 1

    # 读取需要融合的相似性矩阵
    miRNA_cosine_sim = cosine_similarity(M_D)  # 功能相似性
    miRNA_lnc_sim = pd.read_csv('./datasets/similarly/M_lnc.csv', header=None).values   # lncRNA相似性
    miRNA_gene_sim = pd.read_csv('./datasets/similarly/M_gene.csv', header=None).values # gene相似性

    disease_jaccard_sim = calculate_jaccard_similarity(D_M)  # 功能相似性
    disease_lnc_sim = pd.read_csv('./datasets/similarly/D_lnc.csv', header=None).values   # lncRNA相似性
    disease_gene_sim = pd.read_csv('./datasets/similarly/D_gene.csv', header=None).values # gene相似性
    disease_mesh_sim = pd.read_csv('./datasets/similarly/D_mesh.csv', header=None).values # 语义相似性

    M = 0.5 * miRNA_cosine_sim + 0.25 * miRNA_lnc_sim + 0.25 * miRNA_gene_sim
    D = 0.4 * disease_jaccard_sim + 0.2 * disease_mesh_sim + 0.2 * disease_lnc_sim + 0.2 * disease_gene_sim

    M = np.array(M)
    D = np.array(D)

    # 得到邻接矩阵
    M = [[1 if x > 0.12 and (j != i) else 0 for i, x in enumerate(row)] for j, row in enumerate(M)]
    D = [[1 if x > 0.12 and (j != i) else 0 for i, x in enumerate(row)] for j, row in enumerate(D)]

    return torch.tensor(M, dtype=torch.float), torch.tensor(D, dtype=torch.float)


def calculate_jaccard_similarity(matrix):
    size = matrix.shape[0]
    jaccard_sim = np.zeros((size, size))
    for i in range(size):
        for j in range(i, size):
            intersection = np.logical_and(matrix[i], matrix[j]).sum()
            union = np.logical_or(matrix[i], matrix[j]).sum()
            jaccard_sim[i, j] = jaccard_sim[j, i] = intersection / union if union != 0 else 0
    return jaccard_sim


def save_xlsx_data(metrics_per_fold, save_csv_path):
    df_plot = pd.DataFrame([metrics_per_fold['epoch_losses'],
                            metrics_per_fold['epoch_accuracies'],
                            metrics_per_fold['roc_aucs'],
                            metrics_per_fold['accuracies'],
                            metrics_per_fold['precisions'],
                            metrics_per_fold['recalls'],
                            metrics_per_fold['f1s'],
                            metrics_per_fold['fprs'],
                            metrics_per_fold['tprs'],
                            metrics_per_fold['mean_fpr'],
                            metrics_per_fold['mean_tpr'],
                            metrics_per_fold['tprs_'],
                            metrics_per_fold['std_tpr'],
                            metrics_per_fold['pr_precisions'],
                            metrics_per_fold['pr_recalls'],
                            metrics_per_fold['pr_aucs']],

                           index=['Train_Loss', 'Train_accuracy', 'Roc_Auc', 'Accuracy', 'Precision', 'Recall',
                                  'F1_Score', 'fprs', 'tprs', 'mean_fpr', 'mean_tpr', 'tprs_', 'std_tpr',
                                  'pr_precisions', 'pr_recalls', 'pr_aucs'])

    df_plot.to_excel(save_csv_path + 'plot.xlsx', index=True, header=False)
    df = pd.DataFrame(metrics_per_fold['tprs_'])
    df.to_csv(save_csv_path + 'tprs_.csv', index=False)
    df = pd.DataFrame(metrics_per_fold['mean_tpr'])
    df.to_csv(save_csv_path + 'mean_tpr.csv', index=False)
    df = pd.DataFrame(metrics_per_fold['mean_fpr'])
    df.to_csv(save_csv_path + 'mean_fpr.csv', index=False)
    df = pd.DataFrame(metrics_per_fold['epoch_losses'])
    df.to_csv(save_csv_path + 'epoch_losses.csv', index=False)
    df = pd.DataFrame(metrics_per_fold['epoch_accuracies'])
    df.to_csv(save_csv_path + 'epoch_accuracies.csv', index=False)
    df = pd.DataFrame(metrics_per_fold['roc_aucs'])
    df.to_csv(save_csv_path + 'roc_aucs.csv', index=False)
    df = pd.DataFrame(metrics_per_fold['accuracies'])
    df.to_csv(save_csv_path + 'accuracies.csv', index=False)
    df = pd.DataFrame(metrics_per_fold['precisions'])
    df.to_csv(save_csv_path + 'precisions.csv', index=False)
    df = pd.DataFrame(metrics_per_fold['recalls'])
    df.to_csv(save_csv_path + 'recalls.csv', index=False)
    df = pd.DataFrame(metrics_per_fold['f1s'])
    df.to_csv(save_csv_path + 'f1s.csv', index=False)
    df = pd.DataFrame(metrics_per_fold['pr_precisions'])
    df.to_csv(save_csv_path + 'pr_precisions.csv', index=False)
    df = pd.DataFrame(metrics_per_fold['pr_recalls'])
    df.to_csv(save_csv_path + 'pr_recalls.csv', index=False)
    df = pd.DataFrame(metrics_per_fold['pr_aucs'])
    df.to_csv(save_csv_path + 'pr_aucs.csv', index=False)


# 绘制组合图像
def myPlot(metrics_per_fold, save_img_path, save_csv_path):
    # 保存绘图数据
    save_xlsx_data(metrics_per_fold, save_csv_path)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)

    # 设置图表风格
    plt.style.use('classic')

    # 绘制ROC曲线
    for i, (roc_auc, fpr, tpr) in enumerate(zip(metrics_per_fold['roc_aucs'], metrics_per_fold['fprs'],
                                                metrics_per_fold['tprs'])):
        axes[0, 0].plot(fpr, tpr, lw=1, alpha=0.8, label=f'Fold {i + 1} AUC={roc_auc:.4f}')
    axes[0, 0].plot([0, 1], [0, 1], 'k--')
    mean_tpr = metrics_per_fold['mean_tpr']
    mean_tpr[-1] = 1.0
    std_tpr = np.std(metrics_per_fold['tprs_'], axis=0)
    axes[0, 0].set_xlim([-0.05, 1.05])
    axes[0, 0].set_ylim([-0.05, 1.05])
    axes[0, 0].set_xlabel('False_Positive_Rate', fontsize=15)
    axes[0, 0].set_ylabel('True_Positive_Rate', fontsize=15)
    axes[0, 0].set_title('ROC Curves for Each Fold', fontsize=18)
    axes[0, 0].legend(loc="lower right", fontsize=12)

    metrics_titles = ['Roc_Auc', 'Accuracy', 'Precision', 'Recall', 'F1_Score']
    metrics_keys = ['roc_aucs', 'accuracies', 'precisions', 'recalls', 'f1s']

    colors = ['blue', 'green', 'red', 'purple', 'orange']
    plt.style.use('classic')
    for i, (metric_key, title) in enumerate(zip(metrics_keys, metrics_titles)):
        ax = axes.flatten()[i + 1]
        ax.plot(range(1, 6), metrics_per_fold[metric_key], marker='o', linestyle='-', color=colors[i], label=title)
        # 平均线
        mean_value = np.mean(metrics_per_fold[metric_key])
        ax.axhline(mean_value, color=colors[i], linestyle='--', label='Mean')
        # 调整纵轴范围
        min_val, max_val = min(metrics_per_fold[metric_key]), max(metrics_per_fold[metric_key])
        ax.set_ylim(min_val * 0.95, max_val * 1.05)

        ax.set_title(title, fontsize=30)
        ax.set_xlabel('Fold', fontsize=15)
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels([f'Fold {j}' for j in range(1, 6)], fontsize=14)
        ax.set_ylabel(title, fontsize=15)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(fontsize=12)

    fig.savefig(save_img_path + 'Validating_Indexes.png')
    plt.show()

    fig_train, axes = plt.subplots(1, 2, figsize=(24, 12), constrained_layout=True)

    # 绘制LOSS曲线
    for i, fold_loss in enumerate(metrics_per_fold['epoch_losses']):
        axes[0].plot(range(1, len(fold_loss) + 1), fold_loss, label=f'Fold {i + 1}')
    axes[0].set_title('Training_Loss', fontsize=40)
    axes[0].set_xlabel('Epoch', fontsize=30)
    axes[0].set_ylabel('Loss', fontsize=30)
    axes[0].legend(loc='upper right', fontsize=30)

    # 绘制ACC曲线
    for i, fold_acc in enumerate(metrics_per_fold['epoch_accuracies']):
        axes[1].plot(range(1, len(fold_acc) + 1), fold_acc, label=f'Fold {i + 1}')
    axes[1].set_title('Training_Acc', fontsize=40)
    axes[1].set_xlabel('Epoch', fontsize=30)
    axes[1].set_ylabel('Acc', fontsize=30)
    axes[1].legend(loc='lower right', fontsize=30)

    fig_train.savefig(save_img_path + 'Training_Indexes.png')  # 保存图像
    plt.show()

    save_individual_metrics(metrics_per_fold, save_img_path)

# 绘制单个图像
def save_individual_metrics(metrics_per_fold, save_img_path):
    metrics_titles = ['ROC AUC', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Train Accuracy']
    metrics_keys = ['roc_aucs', 'accuracies', 'precisions', 'recalls', 'f1s', 'epoch_accuracies']
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

    plt.style.use('classic')

    # 绘制ROC曲线
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, (roc_auc, fpr, tpr) in enumerate(zip(metrics_per_fold['roc_aucs'], metrics_per_fold['fprs'],
                                                metrics_per_fold['tprs'])):
        ax.plot(fpr, tpr, lw=1, alpha=0.8, label=f'Fold {i + 1} AUC={roc_auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--')
    mean_tpr = metrics_per_fold['mean_tpr']
    mean_tpr[-1] = 1.0
    std_tpr = np.mean(metrics_per_fold['tprs_'], axis=0)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_title('ROC Curves for Each Fold', fontsize=24)
    ax.set_xlabel('False Positive Rate', fontsize=18)
    ax.set_ylabel('True Positive Rate', fontsize=18)
    ax.legend(loc="lower right", fontsize=12)
    fig.savefig(os.path.join(save_img_path, 'ROC_Curves.png'))

    # 绘制PR曲线
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, (precision, recall, pr_auc) in enumerate(
            zip(metrics_per_fold['pr_precisions'], metrics_per_fold['pr_recalls'],
                metrics_per_fold['pr_aucs'])):
        ax.plot(recall, precision, lw=1, alpha=0.8, label=f'Fold {i + 1} PR AUC={pr_auc:.4f}')
    ax.plot([0, 1], [1, 0], 'k--')
    ax.set_title('PR Curves for Each Fold', fontsize=24)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Recall', fontsize=18)
    ax.set_ylabel('Precision', fontsize=18)
    ax.legend(loc="lower left", fontsize=12)
    fig.savefig(os.path.join(save_img_path, 'PR_Curves.png'))


    plt.style.use('classic')
    # 绘制其他评价指标
    for i, (metric_key, title) in enumerate(zip(metrics_keys, metrics_titles)):
        fig, ax = plt.subplots(figsize=(8, 8))
        values = [np.mean(fold) for fold in metrics_per_fold[metric_key]] if metric_key == 'epoch_accuracies' else \
        metrics_per_fold[metric_key]
        ax.plot(range(1, 6), values, marker='o', linestyle='-', color=colors[i % len(colors)], label=title)
        mean_value = np.mean(values)
        ax.axhline(mean_value, color=colors[i % len(colors)], linestyle='--', label='Mean')
        ax.set_ylim(min(values) * 0.95, max(values) * 1.05)
        ax.set_title(title, fontsize=30)
        ax.set_xlabel('Fold', fontsize=16)
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels([f'Fold {j}' for j in range(1, 6)])
        ax.set_ylabel(title, fontsize=16)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(fontsize=12)
        plt.savefig(os.path.join(save_img_path, f'{title.replace(" ", "_")}.png'))
        plt.close()

