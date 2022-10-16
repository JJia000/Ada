import numpy as np
from collections import defaultdict


class word2vec():

    def __init__(self):
        self.n = 16
        self.lr = 0.01
        self.epochs = 50
        self.window = 10

    def generate_training_data(self, corpus):
        """
        得到训练数据
        """

        # defaultdict(int)  一个字典，当所访问的键不存在时，用int类型实例化一个默认值
        word_counts = defaultdict(int)

        # 遍历语料库corpus
        for row in corpus:
            for word in row:
                # 统计每个单词出现的次数
                word_counts[word] += 1

        # 词汇表的长度
        self.v_count = len(word_counts.keys())
        # 在词汇表中的单词组成的列表
        self.words_list = list(word_counts.keys())
        # 以词汇表中单词为key，索引为value的字典数据
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        # 以索引为key，以词汇表中单词为value的字典数据
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))

        training_data = []

        for sentence in corpus:
            sent_len = len(sentence)

            for i, word in enumerate(sentence):

                w_target = self.word2onehot(sentence[i])

                w_context = []

                for j in range(i - self.window, i + self.window):
                    if j != i and j <= sent_len - 1 and j >= 0:
                        w_context.append(self.word2onehot(sentence[j]))

                training_data.append([w_target, w_context])

        return np.array(training_data)

    def word2onehot(self, word):

        # 将词用onehot编码

        word_vec = [0 for i in range(0, self.v_count)]

        word_index = self.word_index[word]

        word_vec[word_index] = 1

        return word_vec

    def train(self, training_data):

        # 随机化参数w1,w2
        self.w1 = np.random.uniform(-1, 1, (self.v_count, self.n))

        self.w2 = np.random.uniform(-1, 1, (self.n, self.v_count))

        for i in range(self.epochs):

            self.loss = 0

            # w_t 是表示目标词的one-hot向量
            # w_t -> w_target,w_c ->w_context
            for w_t, w_c in training_data:
                # 前向传播
                y_pred, h, u = self.forward(w_t)

                # 计算误差
                EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)

                # 反向传播，更新参数
                self.backprop(EI, h, w_t)

                # 计算总损失
                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))

            print('Epoch:', i, "Loss:", self.loss)

    def forward(self, x):
        """
        前向传播
        """

        h = np.dot(self.w1.T, x)

        u = np.dot(self.w2.T, h)

        y_c = self.softmax(u)

        return y_c, h, u

    def softmax(self, x):
        """
        """
        e_x = np.exp(x - np.max(x))

        return e_x / np.sum(e_x)

    def backprop(self, e, h, x):

        d1_dw2 = np.outer(h, e)
        d1_dw1 = np.outer(x, np.dot(self.w2, e.T))

        self.w1 = self.w1 - (self.lr * d1_dw1)
        self.w2 = self.w2 - (self.lr * d1_dw2)

    def word_vec(self, word):

        """
        获取词向量
        通过获取词的索引直接在权重向量中找
        """

        w_index = self.word_index[word]
        v_w = self.w1[w_index]

        return v_w

    def vec_sim(self, word, top_n):
        """
        找相似的词
        """

        v_w1 = self.word_vec(word)
        word_sim = {}

        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_sum = np.dot(v_w1, v_w2)

            # np.linalg.norm(v_w1) 求范数 默认为2范数，即平方和的二次开方
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_sum / theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)

        for word, sim in words_sorted[:top_n]:
            print(word, sim)

    def get_w(self):
        w1 = self.w1
        return w1

'''
# 超参数
settings = {
    'window_size': 2,  # 窗口尺寸 m
    # 单词嵌入(word embedding)的维度,维度也是隐藏层的大小。
    'n': 10,
    'epochs': 50,  # 表示遍历整个样本的次数。在每个epoch中，我们循环通过一遍训练集的样本。
    'learning_rate': 0.01  # 学习率
}

# 数据准备
text = "natural language processing and machine learning is fun and exciting I love Pengyuyan Pengyuyan love me"
# 按照单词间的空格对我们的语料库进行分词
corpus = [[word.lower() for word in text.split()]]
print(corpus)

# 初始化一个word2vec对象
w2v = word2vec()

training_data = w2v.generate_training_data(settings, corpus)

# 训练
w2v.train(training_data)

# 获取词的向量
word = "love"
vec = w2v.word_vec(word)
print(word, vec)

# 找相似的词
w2v.vec_sim("i", 3)
'''

