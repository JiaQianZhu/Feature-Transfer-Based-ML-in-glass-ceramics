import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import hamming_loss
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# 加载数据集
large_data = pd.read_csv('data_lit(clean_449)_13_crystals.csv')  # 原始数据集
large_data = large_data.drop_duplicates()
df_X = large_data.iloc[:, :31]  # 特征
df_y = large_data.iloc[:, 33:]
print(df_X)
df_B = pd.read_csv('las_reorder.csv')  # 小数据集B

# 准备小数据集B的训练数据
X_B = df_B.iloc[:, :3]  # 前三列作为输入特征
y_B = df_B.iloc[:, 3:]  # 第四列开始是标签

# 计算前三列的总和
sum_of_columns = X_B.sum(axis=1)

# 将每个值除以总和，得到相对于总和的比例
X_B_normalized = X_B.div(sum_of_columns, axis=0)

# 将比例乘以100，确保总和为100
X_B_normalized *= 100


# 定义一个简单的特征提取器
class FeatureExtractor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout_prob=0.5):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


# 设置超参数
input_size_B = 3  # 小数据集B输入特征的数量
output_size_B = 7  # 输出特征的数量
hidden_size = 128

learning_rate = 0.001
num_epochs = 200

# 初始化特征提取器
feature_extractor = FeatureExtractor(input_size_B, output_size_B, hidden_size)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()  # 使用均方误差作为损失函数
optimizer = optim.Adam(feature_extractor.parameters(), lr=learning_rate)

# 准备小数据集B的输入和标签
X_train_B = torch.tensor(X_B_normalized.values, dtype=torch.float32)
y_train_B = torch.tensor(y_B.values, dtype=torch.float32)

# 训练特征提取器
for epoch in range(num_epochs):
    # Forward pass
    outputs = feature_extractor(X_train_B)
    loss = criterion(outputs, y_train_B)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 提取特征并将其应用于原始数据集X
X_new_features = feature_extractor(torch.tensor(df_X.iloc[:, :3].values, dtype=torch.float32)).detach().numpy()

# 将新特征与原始数据集的其他特征合并
df_X_with_new_features = pd.concat([df_X, pd.DataFrame(X_new_features)], axis=1)

# 划分训练集和测试集
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(df_X_with_new_features, df_y, test_size=0.2,
                                                                    random_state=42, stratify=df_y)


# 定义神经网络模型
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


# 设置超参数
input_size = 31 + output_size_B  # 原始数据集特征数加上新特征的数量
hidden_size = 64
output_size = 13

learning_rate = 0.001
num_epochs = 200

# 初始化模型
model_all = NeuralNet(input_size, hidden_size, output_size)

# 准备训练数据
X_train_all_input = torch.tensor(X_train_all.iloc[:, :input_size].values, dtype=torch.float32)
y_train_all_input = torch.tensor(y_train_all.values, dtype=torch.float32)

# 定义损失函数和优化器
criterion_all = nn.BCEWithLogitsLoss()  # 二分类问题使用交叉熵损失函数
optimizer_all = optim.Adam(model_all.parameters(), lr=learning_rate)

# 重新训练模型
for epoch in range(num_epochs):
    # Forward pass
    outputs = model_all(X_train_all_input)
    loss = criterion_all(outputs, y_train_all_input)

    # Backward pass and optimization
    optimizer_all.zero_grad()
    loss.backward()
    optimizer_all.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
with torch.no_grad():
    X_test_all_input = torch.tensor(X_test_all.iloc[:, :input_size].values, dtype=torch.float32)
    y_test_all_input = torch.tensor(y_test_all.values, dtype=torch.float32)
    outputs = model_all(X_test_all_input)
    predicted = (outputs >= 0.5).float()
    accuracy = (predicted == y_test_all_input.unsqueeze(1)).float().mean()
    print(f'Accuracy on test set: {accuracy.item():.4f}')

hamming_distance = hamming_loss(y_test_all, predicted)

print(f'Hamming Loss on test set: {hamming_distance:.4f}')

from sklearn.metrics import f1_score

# 计算 F1 分数
f1 = f1_score(y_test_all, predicted, average='micro')

print(f'F1 Score on test set: {f1:.4f}')
