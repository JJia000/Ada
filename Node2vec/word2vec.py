import numpy as np
from collections import defaultdict


class word2vec():

    def __init__(self):
        self.n = 16
        self.lr = 0.01
        self.epochs = 50
        self.window = 10

    def generate_training_data(self, corpus):

        word_counts = defaultdict(int)

        for row in corpus:
            for word in row:
                word_counts[word] += 1

        self.v_count = len(word_counts.keys())
        self.words_list = list(word_counts.keys())
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
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

        word_vec = [0 for i in range(0, self.v_count)]

        word_index = self.word_index[word]

        word_vec[word_index] = 1

        return word_vec

    def train(self, training_data):

        self.w1 = np.random.uniform(-1, 1, (self.v_count, self.n))

        self.w2 = np.random.uniform(-1, 1, (self.n, self.v_count))

        for i in range(self.epochs):

            self.loss = 0

            # w_t -> w_target,w_c ->w_context
            for w_t, w_c in training_data:
                y_pred, h, u = self.forward(w_t)

                EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)

                self.backprop(EI, h, w_t)

                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))

            print('Epoch:', i, "Loss:", self.loss)

    def forward(self, x):

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

        w_index = self.word_index[word]
        v_w = self.w1[w_index]

        return v_w

    def vec_sim(self, word, top_n):

        v_w1 = self.word_vec(word)
        word_sim = {}

        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_sum = np.dot(v_w1, v_w2)

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
