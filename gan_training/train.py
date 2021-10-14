# coding: utf-8
import jittor as jt
from jittor import nn
import numpy as np

class Trainer(object):
    def __init__(self,
                 generator,
                 discriminator,
                 encoder,
                 g_optimizer,
                 d_optimizer,
                 q_optimizer):

        self.generator = generator
        self.discriminator = discriminator
        self.encoder = encoder
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.q_optimizer = q_optimizer

    def generator_trainstep(self, y, z, condition):
        self.generator.train()
        assert (y.size(0) == z.size(0))

        x_fake = self.generator(z, y)
        d_fake = self.discriminator(x_fake, y, condition)
        g_loss = -d_fake.mean()
        # g_loss = nn.binary_cross_entropy_with_logits(d_fake, jt.ones_like(d_fake))

        self.g_optimizer.zero_grad()
        self.g_optimizer.step(g_loss)

        return g_loss.item()

    def discriminator_trainstep(self, x_real, y, z, condition):
        self.discriminator.train()
        # Sampling
        x_fake = self.generator(z, y).detach()
        self.d_optimizer.zero_grad()
        
        jt.start_grad(x_real)
        d_real = self.discriminator(x_real, y, condition)
        d_fake = self.discriminator(x_fake, y, condition)
        # d_loss_real = nn.binary_cross_entropy_with_logits(d_real, jt.ones_like(d_real))        
        # d_loss_fake = nn.binary_cross_entropy_with_logits(d_fake, jt.zeros_like(d_fake))
        d_loss_real = -d_real.mean()
        d_loss_fake = d_fake.mean()

        grad_dout = jt.grad(d_real, x_real)
        grad_dout2 = grad_dout.pow(2)
        reg = grad_dout2.view(x_real.shape[0], -1).sum(1)
        reg_loss = reg.mean()

        d_loss = (d_loss_real + d_loss_fake) / 2.0 + reg_loss 
        d_loss.sync()

        self.d_optimizer.step(d_loss)

        # for p in self.discriminator.parameters():
        #     clamp_(p, - 0.01, 0.01)

        return d_loss_real.item(), d_loss_fake.item(), reg_loss.item()

    def encoder_trainstep(self, y, z, target_embeds):
        assert (y.size(0) == z.size(0))
        # assert (y.size(0) == target_embeds.size(0))

        self.generator.eval()
        self.encoder.train()

        x_fake = self.generator(z, y)
        embeds = self.encoder(x_fake)
        qloss = nn.mse_loss(embeds, target_embeds) * 0.1
        
        self.q_optimizer.zero_grad()
        self.q_optimizer.step(qloss)

        return qloss.item()

def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = jt.grad(d_out, x_in)
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg

def clamp_(var, l, r):
    var.assign(var.maximum(l).minimum(r))
