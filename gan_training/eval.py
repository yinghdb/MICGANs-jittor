import numpy as np
import jittor as jt

from gan_training.utils import compute_purity

class Evaluator(object):
    def __init__(self,
                 generator,
                 encoder,
                 multi_gauss,
                 train_loader,
                 batch_size=64):
        self.generator = generator
        self.encoder = encoder
        self.multi_gauss = multi_gauss
        self.train_loader = train_loader
        self.batch_size = batch_size

    def create_samples(self, z, y=None):
        self.generator.eval()
        batch_size = z.size(0)
        # Parse y
        if y is None:
            raise NotImplementedError()
        elif isinstance(y, int):
            y = jt.ones([batch_size]).long() * y
        # Sample x
        with jt.no_grad():
            x = self.generator(z, y)
        return x

    def compute_purity_score(self):
        predicted_classes = []
        gt_labels = []
        self.encoder.eval()

        with jt.no_grad():
            for x_real, y_gt, _ in self.train_loader:
                x_real = x_real
                embeddings = self.encoder(x_real)
                probs, log_probs = self.multi_gauss.compute_embed_probs(embeddings)
                max_indexes = jt.argmax(log_probs, dim=1)[0]
                predicted_classes.append(max_indexes)
                gt_labels.append(y_gt)

        predicted_classes = jt.contrib.concat(predicted_classes).numpy()
        gt_labels = jt.contrib.concat(gt_labels).numpy()

        score = compute_purity(predicted_classes, gt_labels)

        return score
