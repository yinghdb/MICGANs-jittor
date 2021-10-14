import argparse
import os
import copy
import pprint
from os import path

import cupy
import jittor as jt
from jittor import nn
from jittor import distributions
import numpy as np

from gan_training import utils
from gan_training.train import Trainer
from gan_training.logger import Logger
from gan_training.inputs import get_dataset
from gan_training.eval import Evaluator
from gan_training.config import load_config, build_models, build_optimizers
from clusterers.random_labels import RndClusterer
from clusterers.crp_clusterer import CRPClusterer

import time

jt.flags.use_cuda = 1

# Arguments
parser = argparse.ArgumentParser(
    description='Train a GAN with different regularization strategies.')
parser.add_argument('config', type=str, help='Path to config file.')

args = parser.parse_args()
config = load_config(args.config, 'configs/default.yaml')
out_dir = config['training']['out_dir']

def get_zdist(dist_name, dim):
    # Get distribution
    if dist_name == 'uniform':
        low = -jt.ones(dim)
        high = jt.ones(dim)
        zdist = distributions.Uniform(low, high)
    elif dist_name == 'gauss':
        mu = jt.zeros(dim)
        scale = jt.ones(dim)
        zdist = distributions.Normal(mu, scale)
    else:
        raise NotImplementedError

    # Add dim attribute
    zdist.dim = dim

    return zdist

def main():
    pp = pprint.PrettyPrinter(indent=1)
    pp.pprint(config)
    batch_size = config['training']['batch_size']
    num_k = config['condition']['num_k']

    # Create missing directories
    checkpoint_dir = path.join(out_dir, 'chkpts')
    if not path.exists(out_dir):
        os.makedirs(out_dir)
    if not path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    train_loader = get_dataset(data_dir=config['data']['train_dir'], size=config['data']['img_size']) \
        .set_attrs(batch_size=batch_size, shuffle=True, drop_last=True)
    eval_loader = get_dataset(data_dir=config['data']['train_dir'], size=config['data']['img_size']) \
        .set_attrs(batch_size=batch_size, shuffle=False, drop_last=False)
    
    image_num = eval_loader.dataset.total_len

    # collect ground-truth labels for evaluation
    data_label_ims = np.zeros(image_num, dtype=int)
    for _, real_label, real_index in eval_loader:
        data_label_ims[real_index] = real_label.numpy()
    label_num = len(np.unique(data_label_ims))

    # Create models
    generator, discriminator, encoder, multi_gauss = build_models(config)

    g_optimizer, d_optimizer, q_optimizer = build_optimizers(generator, discriminator, encoder, config)
    
    # Logger
    logger = Logger(log_dir=path.join(out_dir, 'logs'),
                    img_dir=path.join(out_dir, 'imgs'),
                    label_mode_dir=path.join(out_dir, 'label_mode'),
                    mode_label_dir=path.join(out_dir, 'mode_label'),
                    sorted_mode_label_dir=path.join(out_dir, 'sorted_mode_label'))

    # Noise Distribution
    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'])

    # Test samples
    ntest = config['training']['ntest']
    x_test, y_test = utils.get_nsamples(train_loader, ntest)
    z_test = zdist.sample((ntest, config['z_dist']['dim']))

    # Evaluator
    evaluator = Evaluator(
        generator,
        encoder, 
        multi_gauss, 
        train_loader=train_loader,
        batch_size=batch_size)

    # Trainer
    trainer = Trainer(generator,
                      discriminator,
                      encoder,
                      g_optimizer,
                      d_optimizer,
                      q_optimizer)

    # initialization training loop
    print('Start initialization training ...')
    clusterer = RndClusterer(num_k)
    count = 0
    it = 0
    dataiter = iter(train_loader)
    while count < config['initialization']['n_image']:
        try:
            x_real, y, _ = next(dataiter)
        except StopIteration:
            dataiter = iter(train_loader)
            x_real, y, _ = next(dataiter)

        count += batch_size
        it += 1

        # Discriminator updates
        y = clusterer.get_labels(x_real, y)
        z = zdist.sample((batch_size, config['z_dist']['dim']))
        dloss_real, dloss_fake, reg = trainer.discriminator_trainstep(x_real, y, z, condition=False)

        # Generators updates
        y = clusterer.get_labels(x_real, y)
        z = zdist.sample((batch_size, config['z_dist']['dim']))
        gloss = trainer.generator_trainstep(y, z, condition=False)

        # Print stats
        if it % 200 == 0:
            print('init: [it %4d, n %4dk] g_loss = %.4f, d_loss_real = %.4f， d_loss_fake = %.4f, reg=%.4f'
                % (it, count // 1000, gloss, dloss_real, dloss_fake, reg))

        # Sample
        if it % 2000 == 0:
            print('Creating samples...')
            x = evaluator.create_samples(z_test, clusterer.get_labels(x_test, y_test))
            logger.add_imgs(x, 'init/all', count // 1000)

            cat_imgs = []
            for y_inst in range(num_k):
                x = evaluator.create_samples(z_test, y_inst)
                logger.add_imgs(x, 'init/%02d' % y_inst, count // 1000)
                cat_imgs.append(x[:8])

            cat_imgs = jt.contrib.concat(cat_imgs, dim=0)
            logger.add_imgs(cat_imgs, 'init/cat', count // 1000, nrow=8)

    print('Saving backup...')
    outdict = {
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
    }
    jt.save(outdict, os.path.join(checkpoint_dir, "init_GD.pkl"))

    # loaddict = jt.load(os.path.join(checkpoint_dir, "init_GD.pkl"))
    # generator.load_state_dict(loaddict['generator'])
    # discriminator.load_state_dict(loaddict['discriminator'])

    # acrp training loop
    print('Start crp training from epoch')
    clusterer = CRPClusterer(num_k=num_k, multi_gauss=multi_gauss, epoch_1=config['acrp']['cluster_epoch_1'], epoch_2=config['acrp']['cluster_epoch_2'])
    epoch_idx = 0
    while epoch_idx < config['acrp']['n_epoch']:
        epoch_idx += 1

        # training encoder
        print('Start encoder training...')
        count_q = 0
        it = 0
        while count_q < config['acrp']['n_q']:
            count_q += batch_size
            it += 1
            y = clusterer.sample_y(batch_size)
            target_embeds = multi_gauss.get_means(y)
            
            z = zdist.sample((batch_size, config['z_dist']['dim']))
            q_loss = trainer.encoder_trainstep(y, z, target_embeds)
            if it % 100 == 0:
                print('encoder: [epoch %0d n_q %4dk] q_loss = %.4f' % (epoch_idx, count_q//1000, q_loss))
        
        purity_score = evaluator.compute_purity_score()
        print('[epoch %0d] purity: %.4f' % (epoch_idx, purity_score))

        # collect embeddings
        print('Start embedding collection...')
        embedding_ims = jt.zeros((image_num, config['multi_gauss']['embed_dim']))
        with jt.no_grad():
            for x_real, _, index in eval_loader:
                x_real = x_real
                im_embeddings = encoder(x_real)
                embedding_ims[index, :] = im_embeddings
        
        # crp process
        print('Start CRP...')
        mid_results, record_multi_gauss = clusterer.crp(embedding_ims, record=True, dim_reduce=config['multi_gauss']['dim_reduce'])

        # vis count distribution
        y_range, x_ticks = None, None
        for e1 in range(len(mid_results)):
            for e2 in range(len(mid_results[e1])):
                mid_picked_class = mid_results[e1][e2]
                filename=f'{epoch_idx}_{e1}_{e2}'
                y_range, x_ticks = logger.vis_real_data_training_procedure(mid_picked_class, data_label_ims, num_k, label_num, \
                    filename, y_range, x_ticks)
        
        # add prior distribution
        logger.add('distribution', 'distribution', clusterer.distribution, epoch_idx)

        # traininig G & D
        count_gd = 0
        it = 0
        dataiter = iter(train_loader)
        while count_gd < config['acrp']['n_gd']:
            # for x_real, y, index in train_loader:
            try:
                x_real, y, index = next(dataiter)
            except StopIteration:
                dataiter = iter(train_loader)
                x_real, y, index = next(dataiter)

            count_gd += batch_size
            it += 1

            y = clusterer.get_labels(index)
            
            # Discriminator updates
            z = zdist.sample((batch_size, config['z_dist']['dim']))
            dloss_real, dloss_fake, reg = trainer.discriminator_trainstep(x_real, y, z, condition=True)

            # Generators updates
            z = zdist.sample((batch_size, config['z_dist']['dim']))
            gloss = trainer.generator_trainstep(y, z, condition=True)

            # Print stats
            if it % 200 == 0:
                print('GANs: [epoch %0d, n %4dk] g_loss = %.4f, d_loss_real = %.4f, d_loss_fake = %.4f, reg=%.4f'
                    % (epoch_idx, count_gd // 1000, gloss, dloss_real, dloss_fake, reg))

        # Sample
        print('Creating samples...')
        x = evaluator.create_samples(z_test, clusterer.sample_y(z_test.shape[0]))
        logger.add_imgs(x, 'acrp/all', epoch_idx)

        cat_imgs = []
        for y_inst in range(num_k):
            x = evaluator.create_samples(z_test, y_inst)
            logger.add_imgs(x, 'acrp/%02d' % y_inst, epoch_idx)
            cat_imgs.append(x[:8])

        cat_imgs = jt.contrib.concat(cat_imgs, dim=0)
        logger.add_imgs(cat_imgs, 'acrp/cat', epoch_idx, nrow=8)

        # Backup
        print('Saving backup...')
        outdict = {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "encoder": encoder.state_dict(),
            "multi_gauss": multi_gauss.state_dict()
        }
        jt.save(outdict, os.path.join(checkpoint_dir, 'model_%03d.pkl' % epoch_idx))
        logger.save_stats('stats_%03d.p' % epoch_idx)

if __name__ == '__main__':
    main()
