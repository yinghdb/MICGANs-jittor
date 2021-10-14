import matplotlib
matplotlib.use('Agg')

import pickle
import os
import jittor as jt
import copy
import numpy as np
import matplotlib.pyplot as plt

color_list = [
    '#FFDEAD', '#FFD700', '#FFA500', '#66CDAA', '#DCDCDC', '#BDB76B', '#D2B48C', '#20B2AA', 
    '#98FB98', '#FF0000', '#FFE4E1', '#9370DB', '#483D8B', '#A9A9A9', '#00FFFF', '#F0E68C', '#FF00FF', 
    '#F5F5DC', '#EEE8AA', '#FFFFFF', '#9400D3', '#8FBC8F', '#FA8072', '#8B0000', '#FFDAB9', '#FFEBCD', '#C71585', 
    '#F0F8FF', '#00FFFF', '#F0FFF0', '#87CEFA', '#FFEFD5', '#FFF0F5', '#7FFFD4', '#1E90FF', '#FFB6C1', '#8A2BE2', 
    '#FF00FF', '#B22222', '#E9967A', '#8B008B', '#FFFFF0', '#228B22', '#808000', '#DA70D6', '#778899', '#708090']

class Logger(object):
    def __init__(self,
                 log_dir='./logs',
                 img_dir='./imgs',
                 label_mode_dir='./label_mode',
                 mode_label_dir='./mode_label',
                 sorted_mode_label_dir='./sorted_mode_label'):
        self.stats = dict()
        self.log_dir = log_dir
        self.img_dir = img_dir
        self.label_mode_dir = label_mode_dir
        self.mode_label_dir = mode_label_dir
        self.sorted_mode_label_dir = sorted_mode_label_dir

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        if not os.path.exists(label_mode_dir):
            os.makedirs(label_mode_dir)
        if not os.path.exists(mode_label_dir):
            os.makedirs(mode_label_dir)
        if not os.path.exists(sorted_mode_label_dir):
            os.makedirs(sorted_mode_label_dir)


    def add(self, category, k, v, it):
        if category not in self.stats:
            self.stats[category] = {}

        if k not in self.stats[category]:
            self.stats[category][k] = []

        self.stats[category][k].append((it, v))

        k_name = '%s/%s' % (category, k)

    def add_imgs(self, imgs, class_name, it, nrow=8):
        outdir = os.path.join(self.img_dir, class_name)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outfile = os.path.join(outdir, '%08d.png' % it)

        imgs = imgs / 2 + 0.5
        imgs = jt.misc.make_grid(imgs)
        if imgs.shape[0] == 1:
            imgs = jt.misc.repeat(imgs, [3, 1, 1])
        jt.misc.save_image(imgs, outfile, nrow=8)

    def get_last(self, category, k, default=0.):
        if category not in self.stats:
            return default
        elif k not in self.stats[category]:
            return default
        else:
            return self.stats[category][k][-1][1]

    def save_stats(self, filename):
        filename = os.path.join(self.log_dir, filename)
        with open(filename, 'wb') as f:
            pickle.dump(self.stats, f)

    def load_stats(self, filename):
        filename = os.path.join(self.log_dir, filename)
        if not os.path.exists(filename):
            print('Warning: file "%s" does not exist!' % filename)
            return

        try:
            with open(filename, 'rb') as f:
                self.stats = pickle.load(f)
        except EOFError:
            print('Warning: log file corrupted!')

    def vis_real_data_training_procedure(self, mode_ims, label_ims, mode_num, label_num, filename, y_range=None, x_ticks=None):
        label_mode_counts = np.zeros((label_num, mode_num))
        for i in range(len(mode_ims)):
            label_mode_counts[label_ims[i], mode_ims[i]] += 1
        labels = np.array(range(label_num))

        if y_range is None:
            max_mode_num = label_mode_counts.sum(axis=0).max()
            max_range = int((max_mode_num // 1000 + 2) * 1000)
            y_range = np.array(range(0, max_range+1, 1000))
        if x_ticks is None:
            mode_counts = label_mode_counts.sum(axis=0)
            x_ticks = np.argsort(-mode_counts)

        label_mode_counts_sort = label_mode_counts[:, x_ticks]

        out_file1 = os.path.join(self.label_mode_dir, filename)
        out_file2 = os.path.join(self.mode_label_dir, filename)
        out_file3 = os.path.join(self.sorted_mode_label_dir, filename)

        self.vis_real_data_distribution(label_mode_counts, label_mode_counts_sort, labels, out_file1, out_file2, out_file3, y_range, x_ticks)
        
        return y_range, x_ticks

    def vis_real_data_distribution(self, label_mode_counts, label_mode_counts_sort, labels, out_file1, out_file2, out_file3, y_range, x_ticks):
        mode_num = label_mode_counts.shape[1]
        for i in range(mode_num):
            plt.bar(labels, label_mode_counts[:, i], width=0.5, bottom=np.sum(label_mode_counts[:, :i], axis=1), 
                color=color_list[i % mode_num], label=str(i))
        plt.xticks(labels)
        plt.legend(loc=[1, 0])
        plt.savefig(out_file1)
        plt.close('all')

        label_num = len(labels)
        modes = np.array(range(mode_num))
        for i in range(label_num):
            plt.bar(modes, label_mode_counts[i, :], width=0.5, bottom=np.sum(label_mode_counts[:i, :], axis=0), 
                color=color_list[i % label_num], label=str(i))
        plt.xticks(modes)
        plt.yticks(y_range)
        plt.legend(loc=[1, 0])
        plt.savefig(out_file2)
        plt.close('all')

        modes = np.array(range(mode_num))
        for i in range(label_num):
            plt.bar(modes, label_mode_counts_sort[i, :], width=0.5, bottom=np.sum(label_mode_counts_sort[:i, :], axis=0), 
                color=color_list[i % label_num], label=str(i))
        plt.xticks(modes, labels=x_ticks)
        plt.yticks(y_range)
        plt.legend(loc=[1, 0])
        plt.savefig(out_file3)
        plt.close('all')