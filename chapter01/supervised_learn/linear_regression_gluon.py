# 使用gluon 进行线性回归训练
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon


num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += .01 * nd.random_normal(shape=y.shape)

batch_size = 10
dataset = gluon.data.ArrayDataset(X, y)
data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)

# 定义模型
net = gluon.nn.Sequential()

#加入一个Dense层
#输出节点的个数为1
net.add(gluon.nn.Dense(1))

# 初始化模型参数
net.initialize()

# 损失函数
# gluon提供了平方误差函数：
square_loss = gluon.loss.L2Loss()

# 优化
# 随机梯度下降
trainer = gluon.Trainer(
    net.collect_params(), 'sgd', {'learning_rate': 0.1})

epochs = 5
batch_size = 10
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter:
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        total_loss += nd.sum(loss).asscalar()
    print("Epoch %d, average loss: %f" % (e, total_loss/num_examples))

dense = net[0]
print(true_w, dense.weight.data())
print(true_b, dense.bias.data())
