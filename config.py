
class MyConfig():
    def __init__(self):
        self.dataset_name = "amazon"
        self.usertrain_file = "./dataset/" + self.dataset_name + "/traintest/usertrain.txt"
        self.labeltrain_file = "./dataset/" + self.dataset_name + "/traintest/labeltrain.txt"
        self.usertest_file = "./dataset/" + self.dataset_name + "/traintest/usertest.txt"
        self.labeltest_file = "./dataset/" + self.dataset_name + "/traintest/labeltest.txt"

        self.d_model = 16  # Must be the square of a number.
        self.epoch = 1500

        self.p = 2
        self.q = 2
        self.z_dim = 5
        self.learning_rate = 0.0008
