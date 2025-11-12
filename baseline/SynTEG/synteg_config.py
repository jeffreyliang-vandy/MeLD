
class Config(object):
    def __init__(
            self,
    ):  
        self.lstm_dim = 384
        self.n_layer = 3
        self.batch_size = 512
        self.gan_batch_size = 8192
        self.vec_dims = [384, 384, 384]
        self.max_num_visit = 120
        self.max_num_code = 19
        self.max_code_visit = 19
        self.num_code = 225 # the total number of distinct codes
        self.age_space = 30 # check age.npy to derive the number of distinct integer ages from 0 and then add 1.
        self.gender_space = 2  # check gender.npy
        self.country_space = 6  # check country.npy
        self.center_space = 9
        self.birth_space = 30
        self.aidsmode_space = 11
        self.interval_space = 30
        self.Z_DIM = 128
        self.D_DIMS = [768, 768, 768, 768, 768, 768]
        self.G_DIMS = [768, 768, 768, 768, 768, 768, self.num_code + 1]
        self.epoch = 50
        self.lr = 1e-4