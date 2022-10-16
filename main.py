import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import numpy as np
from Node2vec.runNode2vec import runNode2vec
from DCGAN.DCGAN import DCGAN_User
import Tools.tools as tools
from config import MyConfig



if __name__ == '__main__':

    cfg = MyConfig()
    tools.set_seeds()

    # read data
    if os.path.exists(cfg.usertrain_file):
        print("Read saved data...")
        usertrain, labeltrain, usertest, labeltest = tools.read_data(cfg)
    else:
        userid, nP, label = runNode2vec(cfg)
        idtrain, idtest, usertrain, labeltrain, usertest, labeltest= tools.split_dataset(userid, nP, label)
        tools.write_data(usertrain, labeltrain, usertest, labeltest, cfg)

    # extract fake user
    Fake_User = tools.extract_fake(usertrain, labeltrain)

    # train
    G_Fake_User = DCGAN_User(Fake_User, cfg)
    G_Fake_label = [1]*len(G_Fake_User)

    new_usertrain = np.vstack(((np.array(usertrain)), np.array(G_Fake_User)))
    new_labeltrain = np.hstack(((np.array(labeltrain)), np.array(G_Fake_label)))

    # test
    tools.classify(new_usertrain, new_labeltrain, usertest, labeltest)






