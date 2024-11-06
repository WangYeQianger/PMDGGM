import json
import os
import shutil
import time
from torch.nn import DataParallel
import pandas as pd
import torch
from numpy import interp
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import StepLR
from t_model import *
from t_function import *
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, \
    precision_recall_curve, average_precision_score
import warnings

warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f'Using device: {device}')

# 数据预处理
features, edge_index, _, all_samples, labels = load_data()
features = features.to(device)
edge_index = edge_index.to(device)
# 构建图数据对象
data = Data(x=features, edge_index=edge_index, y=labels)

# 将Tensor转换为numpy数组，以便于使用sklearn的StratifiedKFold
samples_numpy = all_samples.numpy()
labels_numpy = labels.numpy()

all_samples = all_samples.to(device)
labels = labels.to(device)


def train(myModel, epochs, learning_rate, save_path, save_img_path, save_csv_path):
    start_time = time.time()
    print("feature_size: ", data.x.shape)
    epochs = epochs
    learning_rate = learning_rate

    random_state = 41
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # 保存当前运行的python文件
    current_file_path = os.path.abspath(__file__)
    current_file_name = os.path.basename(current_file_path)
    target_file_path = save_path + current_file_name
    model_path = "t_model.py"
    target_model_path = save_path + model_path
    function_path = "t_function.py"
    target_function_path = save_path + function_path
    # 复制当前运行的文件到目标目录
    shutil.copy2(current_file_path, target_file_path)
    shutil.copy2(model_path, target_model_path)
    shutil.copy2(function_path, target_function_path)

    tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    # 准备存储交叉验证过程中的评价指标
    metrics_per_fold = {
        'epoch_losses': [],  # 存储每一折的每个epoch的损失
        'epoch_accuracies': [],
        'accuracies': [], 'precisions': [], 'recalls': [], 'f1s': [], 'aucs': [],  # 存储每一折结束后的评价指标
        'roc_aucs': [],
        'fprs': [],
        'tprs': [],
        'mean_fpr': [],
        'mean_tpr': [],
        'tprs_': [],
        'std_tpr': [],
        'pr_precisions': [],  # Add precision for PR curve
        'pr_recalls': [],  # Add recall for PR curve
        'pr_aucs': []
    }

    result = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(samples_numpy, labels_numpy)):
        # 每一折重新初始化模型
        model = myModel(num_features=features.shape[1]).to(device)  # 创建新的模型实例
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss()

        scheduler = StepLR(optimizer, step_size=400, gamma=0.5)

        # 抽取训练集和验证集的样本和标签
        train_samples = all_samples[train_idx]
        val_samples = all_samples[val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        # 根据划分的训练集得出训练的关联关系单向边
        train_edge_index = train_samples[labels[train_idx] == 1].T
        val_edge_index = val_samples[labels[val_idx] == 1].T

        if (fold == 0):
            print("\nBefore add M/D edge_index:")
            print("train_edge_index.shape:", train_edge_index.shape)
            print("val_edge_index.shape:", val_edge_index.shape)

        # 根据 train_edge_index 重新算 M、D 的边并加入边集
        M, D = get_MD_new(train_edge_index.cpu())
        M_edge_index = torch.nonzero(M, as_tuple=False).t().contiguous()
        D_edge_index = torch.nonzero(D, as_tuple=False).t().contiguous()
        D_edge_index += torch.tensor(812)
        # 计算 train_edge_index
        train_edge_index_else = train_edge_index.flip(dims=[0])
        train_edge_index = torch.cat([train_edge_index, train_edge_index_else], dim=1)
        train_edge_index = torch.cat([M_edge_index, D_edge_index, train_edge_index.cpu()], dim=1)
        train_edge_index = torch.tensor(train_edge_index, dtype=torch.long).to(device)

        if (fold == 0):
            print("\ntrain_samples.shape:", train_samples.shape)
            print("val_samples.shape:", val_samples.shape)
            print("\nAfter add M/D edge_index:")
            print("M_edge_index.shape_train: ", M_edge_index.shape)
            print("D_edge_index.shape_train: ", D_edge_index.shape)
            print("train_edge_index.shape:", train_edge_index.shape)

        # 验证集不带权重的边
        M, D = get_MD_new(val_edge_index.cpu())
        M_edge_index = torch.nonzero(M, as_tuple=False).t().contiguous()
        D_edge_index = torch.nonzero(D, as_tuple=False).t().contiguous()
        D_edge_index += torch.tensor(812)
        # 拼接验证集的双向边
        val_edge_index_else = val_edge_index.flip(dims=[0])
        val_edge_index = torch.cat([val_edge_index, val_edge_index_else], dim=1)
        val_edge_index = torch.cat([M_edge_index, D_edge_index, val_edge_index.cpu()], dim=1)
        val_edge_index = torch.tensor(val_edge_index, dtype=torch.long).to(device)

        if (fold == 0):
            print("M_edge_index.shape_val: ", M_edge_index.shape)
            print("D_edge_index.shape_val: ", D_edge_index.shape)
            print("val_edge_index.shape:", val_edge_index.shape)

            # df = pd.DataFrame(train_edge_index.cpu().T)
            # df.to_csv('datasets/train_edge_index_all.csv',index = False)

        epoch_losses = []  # 存储当前折的损失
        epoch_accuracies = []  # 存储当前折的acc

        for epoch in tqdm(range(epochs), desc=f'fold_{fold + 1}', total=epochs, ncols=50):  # 有进度条

            model.train()
            optimizer.zero_grad()
            out, embedding, edge_embedding, attention_out, out_edge_logits = model(
                data.x, train_edge_index.to(device), train_samples)
            out = out.squeeze()
            train_out = out_edge_logits.squeeze().cpu()
            # attention_weights = model.attention.weight.data  # 获取权重
            loss = criterion(train_out, train_labels.cpu())  # cpu和cpu比较，不能经过sigmod函数
            loss.backward()
            optimizer.step()
            # 学习率衰减
            scheduler.step()

            train_preds = (torch.sigmoid(train_out) > 0.5).float()

            train_accuracy = accuracy_score(train_labels.cpu().numpy(), train_preds.detach().numpy())
            # train_accuracy = (train_preds == train_labels.cpu()).float().mean()  # cpu和cpu比较
            epoch_losses.append(loss.item())
            epoch_accuracies.append(train_accuracy)

            # 对训练的最后一轮模型的输出的嵌入特征 embeddings进行可视化
            if (epoch == epochs - 1):

                positive_num = int(torch.sum(train_labels).item())
                # print("edge_embedding.shape: ", edge_embedding.shape)
                # print("positive_num: ", positive_num)
                pca = PCA(n_components=2)
                # 转换为numpy数组，然后使用PCA进行降维
                pca_result = pca.fit_transform(edge_embedding.cpu().detach().numpy())

                # 使用t-SNE进行进一步降维
                tsne = TSNE(n_components=2)
                tsne_result = tsne.fit_transform(pca_result)

                # 可视化t-SNE结果
                # 为miRNA和疾病节点指定两种不同的颜色
                plt.figure(figsize=(10, 8))
                plt.scatter(tsne_result[:positive_num, 0], tsne_result[:positive_num, 1], color='red', label='positive')
                plt.scatter(tsne_result[positive_num:, 0], tsne_result[positive_num:, 1], color='blue',
                            label='negative')

                plt.title('t-SNE edge_embedding of the GCN model')
                plt.xlabel('Principal Component 1')
                plt.ylabel('Principal Component 2')
                plt.legend()  # 添加图例
                # 保存图像
                if (fold == 0):
                    plt.savefig(save_img_path + 'embedding.png')
                plt.show()

        # 每折训练结束后的评估
        model.eval()
        with torch.no_grad():
            out, embedding, edge_embedding, attention_out, out_edge_logits = model(
                data.x, val_edge_index.to(device), val_samples)
            out = out.squeeze()
            val_out = out_edge_logits.squeeze().cpu()

            val_preds = (torch.sigmoid(val_out) > 0.5).float()

            val_labels = val_labels.cpu()  # 标签移动到cpu上

            # roc_curve
            roc_auc = roc_auc_score(val_labels.numpy(), val_out.numpy())  # 这里不能用sigmoid之后的结果
            fpr, tpr, thresholds = roc_curve(val_labels.numpy(), val_out.numpy())  # 这里不能用sigmoid之后的结果
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0

            pr_precision, pr_recall, _ = precision_recall_curve(val_labels.numpy(), val_out.numpy())
            pr_auc = auc(pr_recall, pr_precision)

            accuracy = accuracy_score(val_labels.numpy(), val_preds.numpy())
            precision = precision_score(val_labels.numpy(), val_preds.numpy())
            recall = recall_score(val_labels.numpy(), val_preds.numpy())
            f1 = f1_score(val_labels.numpy(), val_preds.numpy())

            save_pred_path = save_path + 'pred' + '/'
            M = embedding[:812, :]
            D = embedding[812:, :]
            M_D = torch.matmul(M, D.t())
            df = pd.DataFrame(M_D.cpu())
            df.to_csv('{}pred_scores_{}.csv'.format(save_pred_path, fold + 1))

            metrics_per_fold['roc_aucs'].append(roc_auc)
            metrics_per_fold['fprs'].append(fpr)
            metrics_per_fold['tprs'].append(tpr)

            metrics_per_fold['pr_precisions'].append(pr_precision)
            metrics_per_fold['pr_recalls'].append(pr_recall)
            metrics_per_fold['pr_aucs'].append(pr_auc)

            metrics_per_fold['epoch_losses'].append(epoch_losses)
            metrics_per_fold['epoch_accuracies'].append(epoch_accuracies)
            metrics_per_fold['accuracies'].append(accuracy)
            metrics_per_fold['precisions'].append(precision)
            metrics_per_fold['recalls'].append(recall)
            metrics_per_fold['f1s'].append(f1)

            print(f"Fold {fold + 1}")
            print(f"Accuracy: {accuracy:.5f}, "
                  f"Precision: {precision:.5f},"f" Recall: {recall:.5f}, "
                  f"F1: {f1:.5f}, ROCAUC: {roc_auc:.5f}, PRAUC: {pr_auc:.5f}")

            fold_str = (f"Accuracy: {accuracy:.5f}, "
                        f"Precision: {precision:.5f},"f" Recall: {recall:.5f}, "
                        f"F1: {f1:.5f}, ROCAUC: {roc_auc:.5f}, PRAUC: {pr_auc:.5f}")

            result.append(fold_str)

    mean_tpr = np.mean(tprs, axis=0)
    metrics_per_fold['mean_fpr'] = mean_fpr
    metrics_per_fold['mean_tpr'] = mean_tpr
    metrics_per_fold['tprs_'] = tprs

    end_time = time.time()
    # 计算程序运行时间
    execution_time = end_time - start_time
    timestr = "%.3f" % execution_time
    print("运行时间: " + timestr + " s")

    avg_accuracy = np.mean(
        [float(result[0].split(',')[0].split(': ')[1]), float(result[1].split(',')[0].split(': ')[1]),
         float(result[2].split(',')[0].split(': ')[1]), float(result[3].split(',')[0].split(': ')[1]),
         float(result[4].split(',')[0].split(': ')[1])])
    avg_precision = np.mean(
        [float(result[0].split(',')[1].split(': ')[1]), float(result[1].split(',')[1].split(': ')[1]),
         float(result[2].split(',')[1].split(': ')[1]), float(result[3].split(',')[1].split(': ')[1]),
         float(result[4].split(',')[1].split(': ')[1])])
    avg_recall = np.mean([float(result[0].split(',')[2].split(': ')[1]), float(result[1].split(',')[2].split(': ')[1]),
                          float(result[2].split(',')[2].split(': ')[1]), float(result[3].split(',')[2].split(': ')[1]),
                          float(result[4].split(',')[2].split(': ')[1])])
    avg_f1 = np.mean([float(result[0].split(',')[3].split(': ')[1]), float(result[1].split(',')[3].split(': ')[1]),
                      float(result[2].split(',')[3].split(': ')[1]), float(result[3].split(',')[3].split(': ')[1]),
                      float(result[4].split(',')[3].split(': ')[1])])
    avg_rocauc = np.mean([float(result[0].split(',')[4].split(': ')[1]), float(result[1].split(',')[4].split(': ')[1]),
                          float(result[2].split(',')[4].split(': ')[1]), float(result[3].split(',')[4].split(': ')[1]),
                          float(result[4].split(',')[4].split(': ')[1])])
    avg_prauc = np.mean([float(result[0].split(',')[5].split(': ')[1]), float(result[1].split(',')[5].split(': ')[1]),
                         float(result[2].split(',')[5].split(': ')[1]), float(result[3].split(',')[5].split(': ')[1]),
                         float(result[4].split(',')[5].split(': ')[1])])

    print(f"average:")
    print(f"Accuracy: {avg_accuracy:.5f}, "
          f"Precision: {avg_precision:.5f},"f" Recall: {avg_recall:.5f}, "
          f"F1: {avg_f1:.5f}, ROCAUC: {avg_rocauc:.5f}, PRAUC: {avg_prauc:.5f}")

    # 计算参数
    parameters = {
        'learning_rate': learning_rate,
        'model': str(model),
        'epochs': epochs,
        'run_time': timestr,
        'fold1': result[0],
        'fold2': result[1],
        'fold3': result[2],
        'fold4': result[3],
        'fold5': result[4],
        '_avg_': f"Accuracy: {avg_accuracy:.5f}, Precision: {avg_precision:.5f}, Recall: {avg_recall:.5f}, F1: {avg_f1:.5f}, ROCAUC: {avg_rocauc:.5f}, PRAUC: {avg_prauc:.5f}",
        'random_state': random_state,
        'k': 3
    }
    parameters_str = json.dumps(parameters, indent=4)
    file_path = save_path + 'parameters.txt'
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, 'w') as file:
        file.write(parameters_str)

    myPlot(metrics_per_fold, save_img_path, save_csv_path)


if __name__ == '__main__':
    epochs = 1500
    learning_rate = 1e-4

    # save result config
    t = time.localtime()
    time_index = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    save_path = './Result/' + time_index + '/'
    print('\n' + save_path)

    # 创建保存的目录(日期)
    os.mkdir(save_path)
    os.mkdir(save_path + 'plot')
    os.mkdir(save_path + 'pred')
    save_img_path = save_path + 'plot' + '/'  # Result/xxx/plot
    os.mkdir(save_img_path + 'imgs')
    os.mkdir(save_img_path + 'csv')
    save_csv_path = save_img_path + 'csv' + '/'  # plot/csv
    save_img_path = save_img_path + 'imgs' + '/'  # plot/imgs

    train(myModel=Test, epochs=epochs, learning_rate=learning_rate,
          save_path=save_path, save_img_path=save_img_path, save_csv_path=save_csv_path)
