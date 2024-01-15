import random
import collections
import numpy as np
import paddle.fluid as fluid
import parl
from parl.core.fluid import layers
from parl.utils import logger
import turtle as t
from parl.algorithms import DDPG
# import os
# from copy import deepcopy
# from collections import deque
# import matplotlib.pyplot as plt


LEARN_FREQ = 5 # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
ACTOR_LR = 1e-3  # Actor网络的 learning rate  1e-3或者0.01
CRITIC_LR = 1e-3  # Critic网络的 learning rate   1e-3或者0.05
GAMMA = 0.99      # reward 的衰减因子
TAU = 0.001       # 软更新的系数
MEMORY_SIZE = int(1e6)   # 经验池大小 大小为int(1e6)或500000
MEMORY_WARMUP_SIZE = MEMORY_SIZE // 20  # 预存一部分经验之后再开始训练 大小为MEMORY_SIZE // 20 或者 1000  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 256   # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来，原来是256
TRAIN_EPISODE = 3000 # 训练的总episode数
################################# 新增参数
REWARD_SCALE = 0.01   # reward 缩放系数   0.1或者0.01
NOISE = 0.05         # 动作噪声方差


class Paddle():

    def __init__(self):

        self.done = False
        self.reward = 0
        self.hit, self.miss = 0, 0


        #新增参数
        self.min_action = -1.0
        self.max_action = 1.0
        # 新增参数




        # Setup Background

        self.win = t.Screen()
        self.win.title('Paddle')
        self.win.bgcolor('black')
        self.win.setup(width=600, height=600)
        self.win.tracer(0)

        # Paddle

        self.paddle = t.Turtle()
        self.paddle.speed(0)
        self.paddle.shape('square')
        self.paddle.shapesize(stretch_wid=1, stretch_len=5)
        self.paddle.color('white')
        self.paddle.penup()
        self.paddle.goto(0, -275)

        # Ball

        self.ball = t.Turtle()
        self.ball.speed(10)
        self.ball.shape('turtle')
        self.ball.color('blue')
        self.ball.penup()
        self.ball.goto(0, 100)
        self.ball.dx = 3
        self.ball.dy = -3

        # Score

        self.score = t.Turtle()
        self.score.speed(0)
        self.score.color('white')
        self.score.penup()
        self.score.hideturtle()
        self.score.goto(0, 250)
        self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))

        # -------------------- Keyboard control ----------------------

        self.win.listen()
        self.win.onkey(self.paddle_right, 'Right')
        self.win.onkey(self.paddle_left, 'Left')

    # Paddle movement

    def paddle_right(self,action):

        x = self.paddle.xcor()
        if x < 225:
            self.paddle.setx(x+30*abs(action))

    def paddle_left(self,action):

        x = self.paddle.xcor()
        if x > -225:
            self.paddle.setx(x-30*abs(action))

    # ------------------------ AI control ------------------------

    # 0 move left
    # 1 do nothing
    # 2 move right

    def reset(self):

        self.paddle.goto(0, -275)
        self.ball.goto(0, 100)
        return [self.paddle.xcor()*0.01, self.ball.xcor()*0.01, self.ball.ycor()*0.01, self.ball.dx, self.ball.dy]



    def step(self, action):
        action = np.expand_dims(action, 0)
        # print(action)
        action = float(action)  # action作为一个数组有多个元素，此处设定为一个

        self.reward = 0
        self.done = 0

        if action < -0.33:
            self.paddle_left(action)
            self.reward -= 0.05

        if action > 0.33:
            self.paddle_right(action)
            self.reward -= 0.05

        self.run_frame()

        state = [self.paddle.xcor()*0.01, self.ball.xcor()*0.01, self.ball.ycor()*0.01, self.ball.dx, self.ball.dy]
        return self.reward, state, self.done



    def run_frame(self):

        self.win.update()

        # Ball moving

        self.ball.setx(self.ball.xcor() + self.ball.dx)
        self.ball.sety(self.ball.ycor() + self.ball.dy)

        # Ball and Wall collision

        if self.ball.xcor() > 290:
            self.ball.setx(290)
            self.ball.dx *= -1

        if self.ball.xcor() < -290:
            self.ball.setx(-290)
            self.ball.dx *= -1

        if self.ball.ycor() > 290:
            self.ball.sety(290)
            self.ball.dy *= -1

        # Ball Ground contact

        if self.ball.ycor() < -290:
            self.ball.goto(0, 100)
            self.miss += 1
            self.score.clear()
            self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))
            self.reward -= 3
            self.done = True

        # Ball Paddle collision

        if abs(self.ball.ycor() + 250) < 2 and abs(self.paddle.xcor() - self.ball.xcor()) < 55:
            self.ball.dy *= -1
            self.hit += 1
            self.score.clear()
            self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))
            self.reward += 3

#################################################### model模块
class Model(parl.Model):
    def __init__(self, act_dim):
        self.actor_model = ActorModel(act_dim)
        self.critic_model = CriticModel()

    def policy(self, obs): # 链接 ActorModel 下的该方法
        return self.actor_model.policy(obs)

    def value(self, obs, act): # 链接 CriticModel 下的该方法
        return self.critic_model.value(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters() # 基类中的方法，获取参数


class ActorModel(parl.Model): # 演员模型
    def __init__(self, act_dim):
        hid_size = 100

        self.fc1 = layers.fc(size=hid_size, act='relu') # 第一层用 relu 激活
        self.fc2 = layers.fc(size=act_dim, act='tanh') # 第二层用 tanh 激活 -1～1

    def policy(self, obs):  # 输入 obs
        hid = self.fc1(obs)
        means = self.fc2(hid)
        return means # 输出一个 -1～1 的浮点数


class CriticModel(parl.Model): # 评价模型
    def __init__(self):
        hid_size = 100

        self.fc1 = layers.fc(size=hid_size, act='relu') # 第一层用 relu
        self.fc2 = layers.fc(size=1, act=None) # 第二层没有激活函数，线性，因为输出的是 Q 值

    def value(self, obs, act):
        concat = layers.concat([obs, act], axis=1)
        # 沿着第 2 个维度进行拼接，即 行数不变，列数增加
        # 每一个 样本 包含了 obs 和 act
        hid = self.fc1(concat)
        Q = self.fc2(hid)
        Q = layers.squeeze(Q, axes=[1]) # 压缩一维数据
        return Q

#################################################### model模块



#################################################### agent模块
class Agent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim # 状态维度
        self.act_dim = act_dim # 动作维度
        super(Agent, self).__init__(algorithm)

        # 注意：最开始先同步self.model和self.target_model的参数.
        self.alg.sync_target(decay=0)

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program): # 形成预测程序
            # 输入参数定义
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            # 输出参数定义
            self.pred_act = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program): # 形成学习程序
            # 输入参数定义
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(
                name='act', shape=[self.act_dim], dtype='float32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            # 输出参数定义
            _, self.critic_cost = self.alg.learn(obs, act, reward, next_obs,
                                                 terminal)

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0) # 程序输入数据结构要求增维
        act = self.fluid_executor.run(
            self.pred_program, feed={'obs': obs},
            fetch_list=[self.pred_act])[0]
        act = np.squeeze(act)
        # act = np.argmax(act) #尝试
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        # 输入的数据
        act = np.expand_dims(act, -1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('float32'),
            'reward': reward,
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal
        }
        # 运行程序，并取得输出的数据
        critic_cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.critic_cost])[0]
        self.alg.sync_target()
        return critic_cost # 评价网络的 cost
#################################################### agent模块


class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, exp):
        self.buffer.append(exp)
        # print(exp)
        # print(self.buffer)   #存经验没问题

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, done = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)

        return np.array(obs_batch).astype('float32'), \
            np.array(action_batch).astype('float32'), np.array(reward_batch).astype('float32'),\
            np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32')

    def __len__(self):
        return len(self.buffer)



def run_episode(agent, env, rpm):  #玩一轮，一轮有多个steps
    obs = env.reset()
    total_reward = 0
    steps = 0
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))

        # 增加探索扰动, 输出限制在 [-1.0, 1.0] 范围内
        action = np.clip(np.random.normal(action, NOISE), -1.0, 1.0)
        # action 为均值（-1～1），NOISE 为方差，正态分布区值
        # np.clip 限制区间，以免区值超出范围
        # action = float(action)
        reward, next_obs, done = env.step(action) # 交互一步
        # action = [action]  # 方便存入replaymemory
        rpm.append((obs, action, REWARD_SCALE * reward, next_obs, done))
        # print(len(rpm))

        if (len(rpm) > MEMORY_WARMUP_SIZE) and (steps % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            # print("sample成功")
            # print(batch_obs.shape)
            # print(batch_action.shape)
            # print(batch_reward.shape)
            # print(batch_next_obs.shape)
            # print(batch_done.shape)
            # print("ok")
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_done)
            # print("learn了一次")
        obs = next_obs
        total_reward += reward

        if done or steps >= 3000:  #大约能连续接10球就重新开始
            break
    return total_reward


def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        total_reward = 0
        steps = 0
        while True:
            batch_obs = np.expand_dims(obs, axis=0)
            action = agent.predict(batch_obs.astype('float32'))
            action = np.clip(action, -1.0, 1.0)

            steps += 1
            reward, next_obs, done = env.step(action)

            obs = next_obs
            total_reward += reward

            if render:
                env.render()
            if done or steps >= 3000:
                break
        eval_reward.append(total_reward)
    return np.mean(eval_reward)





####################################################
all_train_rewards=[]
all_test_rewards=[]
all_train_steps=[]
all_test_steps=[]


# def draw_process(title,episode,reward,label, color):
#     plt.title(title, fontsize=24)
#     plt.xlabel("episode", fontsize=20)
#     plt.ylabel(label, fontsize=20)
#     plt.plot(episode, reward,color=color,label=label)
#     plt.legend()
#     plt.grid()
#     plt.show()
####################################################


print('1')

env = Paddle()
np.random.seed(0)
action_dim = 1  # actions: 3
obs_dim = 5  # states: 5


# 创建经验池
rpm = ReplayMemory(MEMORY_SIZE)  # DDPG的经验回放池

# 根据parl框架构建agent
# 嵌套Model, DDPG, Agent构建 agent
print('2')
model = Model(act_dim=action_dim)
algorithm = DDPG(model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR) #实例化model
agent = Agent(algorithm, obs_dim=obs_dim, act_dim=action_dim)

print("3")

prev_eval_reward = 50
# 加载模型
# save_path = './DDPG_model.ckpt'
# agent.restore(save_path)

# 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
while len(rpm) < MEMORY_WARMUP_SIZE:
    # print(len(rpm))
    run_episode(agent, env, rpm)


# 开始训练
print('4')
episode = 0
#TRAIN_EPISODE  原先为500
while episode < TRAIN_EPISODE:  # 训练max_episode个回合，test部分不计算入episode数量
    # train part
    for i in range(0, 10):
        total_reward = run_episode(agent, env, rpm)
        all_train_rewards.append(total_reward)
        episode += 1
        # print('Episode %s: total_reward=%.1f' % (episode, total_reward))

    # test part
    eval_reward = evaluate(env, agent, render=False)  # render=True 查看显示效果
    all_test_rewards.append(eval_reward)
    logger.info('episode:{}    test_reward:{}'.format(
        episode, eval_reward))

    if eval_reward > prev_eval_reward:
        prev_eval_reward = eval_reward
        ckpt = 'episode_{}_reward_{}.ckpt'.format(episode, int(eval_reward))
        agent.save('models_dir/'+ckpt)
    # if eval_reward > :
    #     break
# 训练结束，保存模型

save_path = 'models2_dir/DDPG_model2.ckpt'
agent.save(save_path)

ckpt = 'model.ckpt'
agent.restore(ckpt)
evaluate_reward = evaluate(env, agent)
logger.info('Evaluate reward: {}'.format(evaluate_reward))








