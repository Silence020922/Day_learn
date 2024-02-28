from utils import *
from models import *
import argparse
import random
import torch.optim as optim
import uuid
import time
parser = argparse.ArgumentParser()
parser.add_argument('--nlayer',type=int, default=64,help='The number of layers. ')
parser.add_argument('--residual',action="store_true", default=False,help='Residual connect. ')
parser.add_argument('--dropout',type=float, default=0.6,help = 'Dropout. ')
parser.add_argument('--variant',action="store_true",default=False,help='Variant. ')
parser.add_argument('--alpha',type=float,default=0.1,help='Alpha. ')
parser.add_argument('--lamda',type=float,default=0.5,help='Lambda. ')
parser.add_argument('--hidden_size',type=int,default=64,help='Hidden layer size. ')
parser.add_argument('--seed',type=int,default=42,help = 'Random seed.')
parser.add_argument('--wd1',type=float,default=0.01,help = 'wd1 of conv.')
parser.add_argument('--wd2',type=float,default=0.0005,help = 'wd2 of linear.')
parser.add_argument('--lr',type=float,default=0.01,help = 'Learning rate.')
parser.add_argument('--epochs',type=int,default=1500,help='Epoch. ')
parser.add_argument('--patience',type=int,default=100,help='Patience for bad output. ')
parser.add_argument('--test',action='store_true',default=False,help='Test. ')
parser.add_argument('--model',type=str,default='GCNII',help='GCNII, APPNP')
parser.add_argument('--dataset',type=str,default='cora',help='cora, citeseer, pubmed')

args = parser.parse_args()


# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)

checkpt_file = "pretrained/" + uuid.uuid4().hex + ".pt"  # uuid4 基于随机数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset)
if args.model == 'GCNII':
    model = GCNII(features.shape[1],labels.shape[1],args.nlayer,args.hidden_size,args.alpha,args.lamda,args.dropout,args.variant,args.residual).to(device)
elif args.model == 'APPNP':
    model = APPNP(features.shape[1],labels.shape[1],args.nlayer,args.hidden_size,args.alpha,args.dropout).to(device)
features = preprocess_features(features)
adj = preprocess_adj(adj)
labels = preprocess_labels(labels)

optimizer = optim.Adam(
    [
        {"params": model.params1, "weight_decay": args.wd1},
        {"params": model.params2, "weight_decay": args.wd2},
    ],
    lr=args.lr,
)  # learning rate

def train():
    model.train()  # dropout
    optimizer.zero_grad()
    output = model(features, adj)  # Output:label prob
    acc_train = acc(output[idx_train], labels[idx_train].to(device))
    loss_train = fun.nll_loss(output[idx_train], labels[idx_train].to(device)) # cross_entropy 先 softmax 再 nll_loss
    loss_train.backward()  # 反向传播
    optimizer.step()  # 梯度下降进行更新
    return (
        loss_train.item(),
        acc_train,
    )

def val():
    model.eval()
    with torch.no_grad():
        output = model(features, adj)  # Output:label prob
        loss_val = fun.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = acc(output[idx_val], labels[idx_val].to(device))
    return (
        loss_val.item(),
        acc_val,
    )

def test():
    model.load_state_dict(torch.load(checkpt_file))  # 加载best时模型
    model.eval()
    with torch.no_grad():
        output = model(features, adj)  # Output:label prob
        loss_test = fun.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = acc(output[idx_test], labels[idx_test].to(device))
    return (
        loss_test.item(),
        acc_test,
    )

# 训练且保存最佳模型
adj = adj.to(device)
features = features.to(device)
start_time = time.time()
acc_list = list()
for j in range(100):
    print('----------{}%----------'.format(j+1))
    best_loss = float('inf')
    best_acc = 0
    best_epoch = 0
    tolerance = 0
    for i in range(args.epochs):
        loss_train,acc_train = train()
        loss_val, acc_val = val()
        # if (i+1) % 10 == 0:
        #     print('Epoch.:{:04d}, train loss.:{:.3f}, train acc.:{:.3f}, val loss.:{:.3f}, val acc.:{:.3f}'.format(
        #         i+1,loss_train,acc_train,loss_val,acc_val
        #     )) 
        if loss_val < best_loss:
            best_loss = loss_val
            best_acc = acc_val
            best_epoch = i
            torch.save(model.state_dict(),checkpt_file)
            tolerance = 0
        else:
            tolerance += 1

        if tolerance == args.patience: break
        accuracy = best_acc

    if args.test:
        test_loss,test_acc = test()
        accuracy = test_acc
    acc_list.append(accuracy)
acc_mean = np.array(acc_list).mean()
acc_list.append(acc_mean)
np.save('result/{}*_semi.npy'.format(args.dataset),np.array(acc_list))
print("Train cost: {:.4f}s".format(time.time() - start_time))  
print('Test_acc(mean).:{}'.format(acc_mean))
# print('Model {} , loading epoch {:04d}...'.format(args.model,best_epoch))
# print('best loss.:{:.3f},best acc.:{:.3f},total time.:{:.2f}'.format(best_loss,best_acc,time.time()-start_time))
# print('Test' if args.test else 'Val',"acc.:{:.3f}".format(accuracy))








