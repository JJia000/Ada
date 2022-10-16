import os

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import tensorflow.compat.v1 as tf
from sklearn import metrics
import random

SEED = 777


def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_random_seed(seed)
    np.random.seed(seed)


def split_dataset(totaluserid, totaluser, label):
    idtrain = []
    idtest = []
    usertrain = []
    labeltrain = []
    usertest = []
    labeltest = []
    sam = [i for i in range(len(totaluser))]
    tes = random.sample(sam, 1 * (len(totaluser) // 4))
    for i in range(len(tes)):
        sam.remove(tes[i])

    for i in range(len(sam)):
        idtrain.append(totaluserid[sam[i]])
        usertrain.append(totaluser[sam[i]])
        labeltrain.append(int(label[sam[i]]))
    for i in range(len(tes)):
        idtest.append(totaluserid[tes[i]])
        usertest.append(totaluser[tes[i]])
        labeltest.append(int(label[tes[i]]))

    print("train set:{} users, test set:{} users".format(len(usertrain), len(usertest)))

    return idtrain, idtest, usertrain, labeltrain, usertest, labeltest


def classify(x_train, y_train, x_test, y_test):
    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)
    pre_y_test2 = clf.predict(x_test)
    # print("KNeiMetrics:{0}".format((precision_recall_fscore_support(y_test, pre_y_test2, average='weighted'))))
    precision_recall_fscore = precision_recall_fscore_support(y_test, pre_y_test2, average='weighted')
    print("precision:{}".format(precision_recall_fscore[0]))
    print("recall:{}".format(precision_recall_fscore[1]))
    print("f1:{}".format(precision_recall_fscore[2]))
    print("accuracy:{0}".format(accuracy_score(y_test, pre_y_test2)))

    pro_pre = clf.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pro_pre, pos_label=1)
    print("AUC:{0}".format(metrics.auc(fpr, tpr)))


def read_data(cfg):
    usertrain = readuser(cfg.usertrain_file)
    labeltrain = readlabel(cfg.labeltrain_file)
    usertest = readuser(cfg.usertest_file)
    labeltest = readlabel(cfg.labeltest_file)
    return usertrain, labeltrain, usertest, labeltest


def write_data(usertrain, labeltrain, usertest, labeltest, cfg):
    writeuser(usertrain, cfg.usertrain_file)
    writelabel(labeltrain, cfg.labeltrain_file)
    writeuser(usertest, cfg.usertest_file)
    writelabel(labeltest, cfg.labeltest_file)


def extract_fake(usertrain, labeltrain):
    Fake_User = []
    for i in range(len(labeltrain)):
        if labeltrain[i] == int(1):
            Fake_User.append(usertrain[i])
    return Fake_User


def match_userid(cfg):
    outputemb = "./dataset/" + cfg.dataset_name + "/emb/useremb.txt"
    labels = "./dataset/" + cfg.dataset_name + "/labels.txt"

    userid_1, emb = reademb_for_match(outputemb, cfg.d_model)
    userid_2, label = readlabel_for_match(labels)

    label_match = []
    for i in range(len(emb)):
        label_match.append(label[userid_2.index(userid_1[i])])

    return userid_1, emb, label_match


def writeuser(obj, path):
    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))
    with open(path, 'a') as f:
        for i in range(len(obj)):
            for j in range(len(obj[i])):
                f.write(str(obj[i][j]) + " ")
            f.write("\n")

def writelabel(obj, path):
    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))
    with open(path, 'a') as f:
        for i in range(len(obj)):
            f.write(str(obj[i]) + "\n")

def readuser(path):
    with open(path, "r", encoding="utf-8") as f:
        f.seek(0)
        context = []
        for line in f.readlines():
            temp = []
            line = line.split()
            for i in range(len(line)):
                temp.append(float(line[i]))
            context.append(temp)
    return context

def readlabel(path):
    with open(path, "r", encoding="utf-8") as f:
        f.seek(0)
        context = []
        for line in f.readlines():
            context.append(int(line))
    return context


def reademb_for_match(file,K):
    with open(file,"r",encoding="utf-8") as f:
        f.seek(0)
        user1 = []
        emb = []
        one = True
        for line in f.readlines():
            if one == True:
                one = False
                continue
            else:
                line = line.split()
                if line[0][0] != 'B':
                    user1.append(line[0])
                    for i in range(K):
                        line[i+1] = float(line[i+1])
                    emb.append(np.array(line[1:K+1]))
    return user1, emb

def readlabel_for_match(file):
    with open(file ,"r",encoding="utf-8") as f:
        f.seek(0)
        user = []
        label = []
        for line in f.readlines():
            line = line.split()
            user.append(line[0])
            label.append(int(line[1]))

    return user, label
