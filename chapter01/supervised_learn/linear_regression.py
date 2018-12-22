#
# 普通方法线性回归训练模型
#
from mxnet import ndarray as nd
from mxnet import autograd
import matplotlib.pyplot as plt
import random

from common.utils import SGD

num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += .01 * nd.random_normal(shape=y.shape)

# print(X[0], y[0])

# 图像绘制
def graph_print():
    # 第二个特征值 (X[:, 1]) 和目标值 Y 的散点图
    plt.scatter(X[:, 1].asnumpy(),y.asnumpy())
    plt.show()

# graph_print()

batch_size = 10
#每次返回batch_size个随机的样本和对应的目标
def data_iter():
    # 产生一个随机索引
    idx = list(range(num_examples))
    random.shuffle(idx)
    for i in range(0,num_examples,batch_size):
        j = nd.array(idx[i:min(i+batch_size,num_examples)])
        yield nd.take(X,j),nd.take(y,j)

#随机初始化模型参数
w = nd.random_normal(shape=(num_inputs, 1))
b = nd.zeros((1,))
params = [w, b]

#创建梯度
for param in params:
    param.attach_grad()

# 定义模型
# 将输入和模型的权重（w）相乘，再加上偏移（b）
def net(X):
    return nd.dot(X, w) + b

# 损失函数
# 平方误差来衡量预测目标和真实目标之间的差距
def square_loss(yhat, y):
    # 注意这里我们把y变形成yhat的形状来避免矩阵形状的自动转换
    return (yhat - y.reshape(yhat.shape)) ** 2

# 模型函数
def real_fn(X):
    return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2

# 绘制损失随训练次数降低的折线图，以及预测值和真实值的散点图
def plot(losses, X, sample_size=100):
    xs = list(range(len(losses)))
    f, (fg1, fg2) = plt.subplots(1, 2)
    fg1.set_title('Loss during training')
    fg1.plot(xs, losses, '-r')
    fg2.set_title('Estimated vs real function')
    fg2.plot(X[:sample_size, 1].asnumpy(),
             net(X[:sample_size, :]).asnumpy(), 'or', label='Estimated')
    fg2.plot(X[:sample_size, 1].asnumpy(),
             real_fn(X[:sample_size, :]).asnumpy(), '*g', label='Real')
    fg2.legend()
    plt.show()

epochs = 10
learning_rate = .01
niter = 0
losses = []
moving_loss = 0
smoothing_constant = .01

# 训练
for e in range(epochs):
    total_loss = 0

    for data, label in data_iter():
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        SGD(params, learning_rate)
        total_loss += nd.sum(loss).asscalar()

        # 记录每读取一个数据点后，损失的移动平均值的变化；
        niter += 1
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss

        # correct the bias from the moving averages
        est_loss = moving_loss / (1 - (1 - smoothing_constant) ** niter)

        if (niter + 1) % 100 == 0:
            losses.append(est_loss)
            print("Epoch %s, batch %s. Moving avg of loss: %s. Average loss: %f" % (
            e, niter, est_loss, total_loss / num_examples))
            # plot(losses, X)

print(true_b,b)
# print(true_w,w)
