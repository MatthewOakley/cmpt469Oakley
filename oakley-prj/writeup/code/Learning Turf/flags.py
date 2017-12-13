from __future__ import division
import os
from os.path import join as pjoin

import sys

import tensorflow as tf


IMAGE_PIXELS = 256 * 256
NUM_CLASSES = 2


# subjects for training/testing/validation
sub = {'tr':  [''],
       'val': [''],
       'te':  ['']}

flags = tf.app.flags
FLAGS = flags.FLAGS

# Autoencoder Architecture Specific Flags
flags.DEFINE_integer("num_hidden_layers", 2, "Number of hidden layers")

flags.DEFINE_integer('hidden1_units', 512,
                     'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2_units', 64,
                     'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3_units', 25,
                     'Number of units in hidden layer 3.')

flags.DEFINE_integer('image_pixels', IMAGE_PIXELS, 'Total number of pixels')
flags.DEFINE_integer('num_classes', 2, 'Number of classes')

flags.DEFINE_float('pre_layer1_learning_rate', 0.001,
                   'Initial learning rate.')
flags.DEFINE_float('pre_layer2_learning_rate', 0.0001,
                   'Initial learning rate.')
flags.DEFINE_float('pre_layer3_learning_rate', 0.0001,
                   'Initial learning rate.')

flags.DEFINE_float('noise_1', 0.0, 'Rate at which to set pixels to 0')
flags.DEFINE_float('noise_2', 0.0, 'Rate at which to set pixels to 0')
flags.DEFINE_float('noise_3', 0.0, 'Rate at which to set pixels to 0')

# Constants
flags.DEFINE_integer('seed', 1234, 'Random seed')
flags.DEFINE_integer('image_size', 256, 'Image square size')

flags.DEFINE_integer('batch_size', 50,
                     'Batch size. Must divide evenly into the dataset sizes.')

flags.DEFINE_float('supervised_learning_rate', 0.1,
                   'Supervised initial learning rate.')

flags.DEFINE_integer('pretraining_epochs', 40,
                     "Number of training epochs for pretraining layers")
flags.DEFINE_integer('finetuning_epochs', 25,
                     "Number of training epochs for "
                     "fine tuning supervised step")

flags.DEFINE_float('zero_bound', 1.0e-9,
                   'Value to use as buffer to avoid '
                   'numerical issues at 0')
flags.DEFINE_float('one_bound', 1.0 - 1.0e-9,
                   'Value to use as buffer to avoid numerical issues at 1')

flags.DEFINE_float('flush_secs', 120, 'Number of seconds to flush summaries')

# Directories
flags.DEFINE_string('data_dir', '../Faces/allFaces/*.jpg',
                    'Directory to put the training data.')

flags.DEFINE_string('summary_dir', './summaries',
                    'Directory to put the summary data')

flags.DEFINE_string('chkpt_dir', './chkpts',
                    'Directory to put the model checkpoints')

# TensorBoard
flags.DEFINE_boolean('no_browser', False,
                     'Whether to start browser for TensorBoard')

# Python
flags.DEFINE_string('python', sys.executable,
                    'Path to python executable')
