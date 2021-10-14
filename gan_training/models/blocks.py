import jittor as jt
from jittor import nn



class LatentEmbeddingConcat(jt.Module):
    ''' projects class embedding onto hypersphere and returns the concat of the latent and the class embedding '''

    def __init__(self, nlabels, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(nlabels, embed_dim)

    def execute(self, z, y):
        assert (y.size(0) == z.size(0))
        yembed = self.embedding(y)
        yembed = yembed / jt.norm(yembed, p=2, dim=1, keepdim=True)
        yz = jt.contrib.concat([z, yembed], dim=1)
        return yz

class LatentEmbeddingAdd(jt.Module):
    ''' projects class embedding onto hypersphere and returns the concat of the latent and the class embedding '''

    def __init__(self, nlabels, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(nlabels, embed_dim)

    def execute(self, z, y):
        assert (y.size(0) == z.size(0))
        yembed = self.embedding(y)
        assert (z.size(1) == yembed.size(1))
        yz = z + yembed
        return yz

class Identity(jt.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def execute(self, inp, *args, **kwargs):
        return inp


class LinearConditionalMaskLogits(jt.Module):
    ''' runs activated logits through fc and masks out the appropriate discriminator score according to class number'''

    def __init__(self, nc, nlabels):
        super().__init__()
        self.fc = nn.Linear(nc, nlabels)

    def execute(self, inp, y=None, take_best=False, get_features=False):
        out = self.fc(inp)
        if get_features: return out

        if not take_best:
            y = y.view(-1)
            index = jt.arange(out.size(0), dtype="int64")
            return out[index, y]
        else:
            # high activation means real, so take the highest activations
            best_logits, _ = out.max(dim=1)
            return best_logits


class LinearUnconditionalLogits(jt.Module):
    ''' standard discriminator logit layer '''

    def __init__(self, nc):
        super().__init__()
        self.fc = nn.Linear(nc, 1)

    def execute(self, inp, y, take_best=False):
        assert (take_best == False)

        out = self.fc(inp)
        return out.view(out.size(0))

