import torch
from collections import namedtuple
import numpy as np
import torch.nn as nn
import torch.optim as optim
from GraphSAGE import GraphSage
from Data import CoraData
from GraphSAGE import multi_sampling

# 数据准备
Data = namedtuple('Data', ['x', 'y', 'adjacency_dict',
                           'train_mask', 'val_mask', 'test_mask']) # 命名元组
data = CoraData().data  
x = data.x / data.x.sum(1, keepdims=True)  # 归一化数据，使得每一行和为1
train_index = np.where(data.train_mask)[0] # 设置训练集
train_label = data.y[train_index]
test_index = np.where(data.test_mask)[0] # 设置测试集

# 模型初始化
INPUT_DIM = 1433   # 输入维度

# Note: 采样的邻居阶数需要与GCN的层数保持一致
HIDDEN_DIM = [128, 7]   # 隐藏单元节点数
NUM_NEIGHBORS_LIST = [10, 10]   # 每阶采样邻居的节点数
assert len(HIDDEN_DIM) == len(NUM_NEIGHBORS_LIST)

BTACH_SIZE = 16     # 批处理大小
EPOCHS = 20
NUM_BATCH_PER_EPOCH = 20    # 每个epoch循环的批次数
LEARNING_RATE = 0.01    # alpha
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(torch.cuda.is_available())
model = GraphSage(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                  num_neighbors_list=NUM_NEIGHBORS_LIST).to(DEVICE)
# print(model)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

# Train and Test
def train():
    model.train() # 启用BN层
    for e in range(EPOCHS): # 进入每个epoch
        for batch in range(NUM_BATCH_PER_EPOCH): # 选择 batch
            batch_src_index = np.random.choice(train_index, size=(BTACH_SIZE,))
            batch_src_label = torch.from_numpy(train_label[batch_src_index]).long().to(DEVICE) 
            batch_sampling_result = multi_sampling(batch_src_index, NUM_NEIGHBORS_LIST, data.adjacency_dict) # 得到的为节点标号，还需要进一步取其特征
            batch_sampling_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in batch_sampling_result]
            batch_train_logits = model(batch_sampling_x)
            loss = criterion(batch_train_logits, batch_src_label) # 交叉信息熵计算损失函数
            optimizer.zero_grad() # 梯度归零
            loss.backward()  # 反向传播计算参数的梯度
            optimizer.step()  # 使用优化方法进行梯度更新
            print("Epoch {:03d} Batch {:03d} Loss: {:.4f}".format(e, batch, loss.item()))
        test()


def test():
    model.eval()
    with torch.no_grad():
        test_sampling_result = multi_sampling(test_index, NUM_NEIGHBORS_LIST, data.adjacency_dict)
        test_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in test_sampling_result]
        test_logits = model(test_x)
        test_label = torch.from_numpy(data.y[test_index]).long().to(DEVICE)
        predict_y = test_logits.max(1)[1]
        accuarcy = torch.eq(predict_y, test_label).float().mean().item()
        print("Test Accuracy: ", accuarcy)

# train()