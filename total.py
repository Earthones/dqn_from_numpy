import numpy as np
import gym
import copy
# np.random.seed(5)

class linear(object):
    # y = xw + b    p*n = p*m * m*n + 1*n      p:batch_size   m:in_features    n:out_features
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.w = np.random.normal(0, np.sqrt(2/4) ,(in_features, out_features)) #这两种训练不起来，参数太大！
        self.b = np.zeros((1, out_features))

    def forward(self, x):  # 前向传播
        self.x = x
        return np.dot(self.x, self.w) + self.b  # numpy广播

    def backward(self, back_grad):  # 反向传播 计算梯度
        self.w_grad = np.dot(self.x.T, back_grad)
        self.x_grad = np.dot(back_grad, self.w.T)
        self.b_grad = np.sum(back_grad, axis=0)
        return self.x_grad

    def update(self, lr=0.001):  # 更新参数 mini-batch GD
        self.w -= lr * self.w_grad
        self.b -= lr * self.b_grad

    def load_param(self, w, b):  # 参数加载
        assert self.w.shape == w.shape
        assert self.b.shape == b.shape
        self.w = w
        self.b = b

    def save_param(self):  # 参数保存
        return self.w, self.b


class sigmoid(object):
    def s(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        self.x = x
        return self.s(x)

    def backward(self, back_grad):
        return back_grad * self.s(self.x) * (1 - self.s(self.x))


class relu(object):
    def forward(self, x):
        self.x = x
        output = np.maximum(0, x)
        return output

    def backward(self, back_grad):
        top_grad = back_grad
        top_grad[self.x <= 0] = 0

        return top_grad

class model(object):
    def __init__(self):
        self.l1 = linear(4, 50)
        self.r1 = sigmoid()
        self.s1 = sigmoid()
        self.l2 = linear(50, 30)
        self.r2 = relu()
        self.s2 = sigmoid()
        self.l3 = linear(30, 2)
        self.layers = [self.l1, self.r1, self.l2, self.r2, self.l3]  #可以进行更改,改变模型架构

        # self.velocity 用于 momentum sgd
        self.velocity = {layer: {'w': np.zeros_like(layer.w), 'b': np.zeros_like(layer.b)} for layer in self.layers if hasattr(layer, 'w')}
        #self.m 和self.v用于adam优化算法
        self.m = {}
        self.v = {}
        for layer in self.layers:
            if hasattr(layer, 'w'):
                self.m[layer] = {'w': np.zeros_like(layer.w), 'b': np.zeros_like(layer.b)}
                self.v[layer] = {'w': np.zeros_like(layer.w), 'b': np.zeros_like(layer.b)}

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def cal_grad(self, b_a, b_r, q_, q, output):  #200, 200, 200, 200, 200*2
        # loss_function = 1/2 * (r + target_q - q)**2
        backgrad = np.zeros_like(output)
        for i in range(output.shape[0]):
            backgrad[i, b_a[i]] = q[i] - b_r[i] - q_[i] # 只对有用的梯度进行计算
        backgrad /= output.shape[0]  # 实际是多个的梯度 大小应该是200*2 不知道这里要不要除 output.shape[0]
        self.last_backgrad = backgrad

    def backward(self):
        back_grad = self.last_backgrad
        for layer in self.layers[::-1]:  # 梯度往回传
            back_grad = layer.backward(back_grad)

    def update(self,count, mode="adam", lr=0.001):  # 适当的学习率
        if mode == "sgd":
            for layer in self.layers:
                if hasattr(layer, "update"):
                    universal_update = getattr(layer, "update")
                    universal_update(lr)
        if mode == "momentum sgd":
            momentum = 0.9
            for layer in self.layers:
                if hasattr(layer, "update"):
                    v_w = self.velocity[layer]['w'] # 更新速度
                    v_b = self.velocity[layer]['b']
                    v_w = momentum * v_w + lr * layer.w_grad
                    v_b = momentum * v_b + lr * layer.b_grad
                    # 更新参数
                    layer.w -= v_w
                    layer.b -= v_b
                    # 存储更新后的速度
                    self.velocity[layer]['w'] = v_w
                    self.velocity[layer]['b'] = v_b
        if mode == "adam":
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            t = count
            for layer in self.layers:
                if hasattr(layer, 'w'):
                    t += 1  # 更新时间步
                    # 获取梯度
                    grad_weights = layer.w_grad
                    grad_biases = layer.b_grad

                    # 更新动量和速度
                    self.m[layer]['w'] = self.beta1 * self.m[layer]['w'] + (1 - self.beta1) * grad_weights
                    self.v[layer]['w'] = self.beta2 * self.v[layer]['w'] + (1 - self.beta2) * (grad_weights ** 2)
                    self.m[layer]['b'] = self.beta1 * self.m[layer]['b'] + (1 - self.beta1) * grad_biases
                    self.v[layer]['b'] = self.beta2 * self.v[layer]['b'] + (1 - self.beta2) * (grad_biases ** 2)
                    # 计算偏差修正后的估计值
                    m_hat_weights = self.m[layer]['w'] / (1 - self.beta1 ** t)
                    v_hat_weights = self.v[layer]['w'] / (1 - self.beta2 ** t)
                    m_hat_biases = self.m[layer]['b'] / (1 - self.beta1 ** t)
                    v_hat_biases = self.v[layer]['b'] / (1 - self.beta2 ** t)
                    # 更新权重和偏置
                    layer.w -= lr * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)
                    layer.b -= lr * m_hat_biases / (np.sqrt(v_hat_biases) + self.epsilon)

    def deepcopy_model(self, target_net):
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "update"): #复制权重和偏置
                target_net.layers[i].w = np.copy(layer.w)
                target_net.layers[i].b = np.copy(layer.b)

    def save_model(self, param_dir):
        print('Saving parameters to file ' + param_dir)
        params = [] #[(1,2),()...]
        for layer in (self.layers):
            if hasattr(layer, "update"):
                params.append(layer.save_param())
        np.save(param_dir, params)

    def load_model(self, param_dir):
        print('Loading parameters from file ' + param_dir)
        params = np.load(param_dir,allow_pickle=True)
        i = 0
        for layer in (self.layers):
            if hasattr(layer, "update"):
                layer.load_param(*params[i])
                i+=1
class replay_buffer(object):
    def __init__(self, buffer_size=1000, batch_size=200):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = np.zeros((self.buffer_size, 10))  # 同一个numpy对象 类型相同
        self.store_count = 0

    def add(self, s, a, s_, r):
        self.buffer[self.store_count % self.buffer_size][0:4] = s
        self.buffer[self.store_count % self.buffer_size][4:5] = a
        self.buffer[self.store_count % self.buffer_size][5:9] = s_
        self.buffer[self.store_count % self.buffer_size][9:10] = r
        self.store_count += 1

    def sample(self):
        index = np.random.randint(0, self.buffer_size, size=self.batch_size) #按随机数之后的顺序取
        index1 = np.random.randint(0, self.buffer_size - self.batch_size - 1) #随机取一个batch
        b_s = self.buffer[index1: (index1 + self.batch_size), 0:4]
        b_a = self.buffer[index1: (index1 + self.batch_size), 4:5]
        b_s_ = self.buffer[index1: (index1 + self.batch_size), 5:9]
        b_r = self.buffer[index1: (index1 + self.batch_size), 9:10]
        return b_s, b_a, b_s_, b_r

    def __len__(self):
        return self.store_count


def train():
    env = gym.make('CartPole-v1')
    env = env.unwrapped
    net = model()
    target_net = copy.deepcopy(net)
    buffer_size = 2000 # 总的样本大小
    batch_size = 500  # 一次学习的样本数 200
    learn_time = 0 # net总共的学习次数
    update_time = 20  # 每20次更新一次old网络参数
    target_learn_time = 0 #target_net总共的学习次数
    gamma = 0.85  # 折扣因子
    decline = 0.6  # 衰减系数
    buffer = replay_buffer(buffer_size, batch_size)
    r_total_average = 0 # 所有轮次的reward的平均值
    r_list = [0] * 20 #计算最近20轮的ep_r (epoch reward)
    best_reward = 0 # 记录最好的r_list的平均值
    count = 0 # 计算进行的轮次
    for j in range(2000):
        ep_r = 0 # 一轮的reward
        s = env.reset()
        while True:
            epsilon = 0.1
            if np.random.randint(0,100) < 100*(decline**learn_time):
                a = np.random.randint(0,2)
            else:
                out = net.forward(s)
                a = np.argmax(out)
            s_, rr, done, info = env.step(a) # rr表示环境给的奖励
            x, theta = s_[0], s_[2]
            r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8 #位置于正负0.48内都为正
            r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5 #限定位置于2.3度内奖励值为正
            r = r1 + r2 # r自己定义的奖励 效果更好
            buffer.add(s, a, s_, r)
            s = s_
            # 当收集的buffer大于buffer_size时开始学习
            if len(buffer) > buffer_size:
                env.render()
                if learn_time == 0:
                    print("开始学习")
                # 更新targetnetwork
                if learn_time % update_time == 0: # 每过update_time次就更新目标网络
                    target_net = copy.deepcopy(net)
                    target_learn_time += 1
                    if target_learn_time % 20 == 0: # 每更新20次目标网络打印
                        print(f"更新了{learn_time}次net,更新了{learn_time//update_time}次target_net")

                b_s, b_a, b_s_, b_r = buffer.sample()
                b_a = b_a.astype(np.int32)
                b_a = b_a.reshape(len(b_a))
                b_r = b_r.reshape(b_r.shape[0]) #batch*1 -> batch
                output = net.forward(b_s)  # b*2  取对应q值
                q = np.zeros(len(b_a))
                for i in range(len(output)):
                    q[i] = output[i, b_a[i]]
                target_output = target_net.forward(b_s_)  # b*2
                q_ = np.zeros(len(b_a))
                for i in range(len(output)):
                    q_[i] = np.max(target_output[i])
                loss = 1 / 2 * np.sum((b_r + gamma * q_ - q) ** 2) / batch_size
                net.cal_grad(b_a, b_r, q_, q, output)
                net.backward()
                net.update(count, mode="adam", lr=0.01) # count供adam使用
                learn_time += 1
                ep_r += rr
                if done:
                    count += 1
                    r_total_average = (r_total_average * count + ep_r) / (count + 1)
                    r_list[j % 20] = ep_r
                    print('episode: ', j,
                          'current r: ', round(ep_r, 2),
                          'average r of r_list: ', round(sum(r_list)/len(r_list), 2),
                          "total average r", round(r_total_average, 2),
                          "loss", round(loss, 2))
                    if ep_r > 1000:
                        if sum(r_list) > best_reward:
                            best_reward = sum(r_list)/len(r_list)
                            print(f"current best reward:{best_reward}")
                            net.save_model(f"./model_average_reward:{int(best_reward)}.npy")
                    break

            if done:
                break

def test(dir):
    net = model()
    net.load_model(dir)
    env = gym.make('CartPole-v1')
    env = env.unwrapped
    for j in range(500):
        ep_r = 0
        s = env.reset()
        while True:
            env.render()
            out = net.forward(s) # 怎么传的
            a = np.argmax(out) # axis没写
            s_, rr, done, info = env.step(a)
            ep_r += rr
            s = s_
            if done:
                print("当前epoch的reward:",ep_r)
                break


if __name__ == '__main__':
    # test("./model_para.npy")
    train()
