import jittor as jt

class RndClusterer():
    def __init__(self, num_k):
        super().__init__()
        self.num_k = num_k

    def sample_y(self, batch_size):
        return jt.randint(low=0, high=self.num_k, shape=[batch_size]).long()

    def get_labels(self, x, y):
        return jt.randint(low=0, high=self.num_k, shape=y.shape).long()

    def get_one_label(self):
        return  jt.randint(low=0, high=self.num_k, shape=[1]).long()