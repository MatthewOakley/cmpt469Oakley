"""Functions for downloading and reading MNIST data."""
from __future__ import division
from __future__ import print_function

import gzip

from six.moves import urllib
from six.moves import range  # pylint: disable=redefined-builtin
from flags import FLAGS
import os
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

from PIL import Image
import glob
import numpy as np


def maybe_download(filename, work_directory):
  """Download the data from Yann's website, unless it's already here."""
  if not os.path.exists(work_directory):
    os.mkdir(work_directory)
  filepath = os.path.join(work_directory, filename)
  if not os.path.exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  return filepath


def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)


def load_images(f, sub):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
  Args:
    f: A string that matches a specific pattern of files in a directory.
    sub: A list of strings indicating the subjects to process.
  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].
  """
  print('Loading images ', f, sub)
  filelist = glob.glob(f)
  
  res=[]
  for aFile in filelist:
    for s in sub:
      if s in aFile:
        res.append(aFile)

  filelist = res

  # open the first to get dimensions
  tmp = np.array(Image.open(filelist[0]));
  num_images = len(filelist)
  rows = tmp.shape[0]
  cols = tmp.shape[1]
  print("Number of images: %d" % (num_images))
  print("Number of rows: %d" % (rows))
  print("Number of cols: %d" % (cols))

  data = np.array([np.array(Image.open(fname)) for fname in filelist])
  data = data.reshape(num_images, rows, cols, 1)

  print("Dataset array size: ", data.shape)

  return data


def extract_images(filename):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  print('\nExtracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError(
          'Invalid magic number %d in MNIST image file: %s' %
          (magic, filename))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot



def extract_labels(f, sub, one_hot=False, num_classes=10):
  """Extract the labels into a 1D uint8 numpy array [index].
  Args:
    f: A string that matches a specific pattern of files in a directory.
    sub: A list of strings indicating the subjects to process.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.
  Returns:
    labels: a 1D uint8 numpy array.
  """
  print('Extracting labels from data in ', f, sub)
  filelist = glob.glob(f)

  res=[]
  for aFile in filelist:
    for s in sub:
      if s in aFile:
        res.append(aFile)

  filelist = res


  labels = np.array([np.array(fn[-10], dtype=np.uint8) for fn in filelist])
  #labels -= 1

  if one_hot:
    return dense_to_one_hot(labels, num_classes)

  return labels




class DataSet(object):

  def __init__(self, images, labels):
    assert images.shape[0] == labels.shape[0], (
        "images.shape: %s labels.shape: %s" % (images.shape,
                                               labels.shape))
    self._num_examples = images.shape[0]

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    assert images.shape[3] == 1
    images = images.reshape(images.shape[0],
                            images.shape[1] * images.shape[2])
    # Convert from [0, 255] -> [0.0, 1.0].
    images = images.astype(np.float32)
    images = np.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


class DataSetPreTraining(object):

  def __init__(self, images):
    self._num_examples = images.shape[0]

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    assert images.shape[3] == 1
    images = images.reshape(images.shape[0],
                            images.shape[1] * images.shape[2])
    # Convert from [0, 255] -> [0.0, 1.0].
    images = images.astype(np.float32)
    images = np.multiply(images, 1.0 / 255.0)
    self._images = images
    self._images[self._images < FLAGS.zero_bound] = FLAGS.zero_bound
    self._images[self._images > FLAGS.one_bound] = FLAGS.one_bound
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._images[start:end]


def read_data_sets(dataset_dir_pattern,
                   tr_sub, te_sub=[], val_sub=[],
                   one_hot=False):
  class DataSets(object):
    pass
  data_sets = DataSets()

  train_images = load_images(dataset_dir_pattern, tr_sub)
  train_labels = extract_labels(dataset_dir_pattern, tr_sub, one_hot=one_hot)

  if (not te_sub) and (not val_sub):
    pct_tr = 0.5  
    pct_te = 0.25
    pct_val = 0.25
    idx = np.arange(train_labels.shape[0])
    np.random.shuffle(idx)
    idx_tr = idx[0:np.int(len(idx)*pct_tr)]
    idx_te = idx[np.int(len(idx)*pct_tr):np.int(len(idx)*(pct_tr+pct_te))]
    idx_val = idx[np.int(len(idx)*(pct_tr+pct_te)):]

    test_images = train_images[idx_te, :, :, :]
    if one_hot:
      test_labels = train_labels[idx_te, :]
    else:
      test_labels = train_labels[idx_te]
    
    validation_images = train_images[idx_val, :, :, :]
    if one_hot:
      validation_labels = train_labels[idx_val, :] 
    else:
      validation_labels = train_labels[idx_val] 
    
    train_images = train_images[idx_tr, :, :, :]
    if one_hot:
      train_labels = train_labels[idx_tr, :]
    else:
      train_labels = train_labels[idx_tr]
  
  else:
    test_images = load_images(dataset_dir_pattern, te_sub)
    test_labels = extract_labels(dataset_dir_pattern, te_sub, one_hot=one_hot)

    validation_images = load_images(dataset_dir_pattern, val_sub)
    validation_labels = extract_labels(dataset_dir_pattern, 
                                       val_sub, 
                                       one_hot=one_hot)

  print(train_images.shape) 
  print(train_labels.shape) 
  print(test_images.shape) 
  print(test_labels.shape) 
  print(validation_images.shape) 
  print(validation_labels.shape) 
  
  data_sets.train = DataSet(train_images, train_labels)
  data_sets.validation = DataSet(validation_images, validation_labels)
  data_sets.test = DataSet(test_images, test_labels)

  return data_sets


def read_data_sets_pretraining(dataset_dir_pattern,
                               tr_sub, te_sub=[], val_sub=[],
                               one_hot=False):
  print(tr_sub)
  print(te_sub)
  print(val_sub)
  class DataSets(object):
    pass
  data_sets = DataSets()

  train_images = load_images(dataset_dir_pattern, tr_sub)

  if (not te_sub) and (not val_sub):
    pct_tr = 0.5  
    pct_te = 0.25
    pct_val = 0.25
    idx = np.arange(train_images.shape[0])
    np.random.shuffle(idx)
    idx_tr = idx[0:np.int(len(idx)*pct_tr)]
    idx_te = idx[np.int(len(idx)*pct_tr):np.int(len(idx)*(pct_tr+pct_te))]
    idx_val = idx[np.int(len(idx)*(pct_tr+pct_te)):]

    test_images = train_images[idx_te, :, :, :]
    
    validation_images = train_images[idx_val, :, :, :]
    
    train_images = train_images[idx_tr, :, :, :]
  
  else:
    test_images = load_images(dataset_dir_pattern, te_sub)

    validation_images = load_images(dataset_dir_pattern, val_sub)

  print(train_images.shape) 
  print(test_images.shape) 
  print(validation_images.shape) 
  
  data_sets.train = DataSetPreTraining(train_images)
  data_sets.validation = DataSetPreTraining(validation_images)
  data_sets.test = DataSetPreTraining(test_images)

  return data_sets



def _add_noise(x, rate):
  x_cp = np.copy(x)
  pix_to_drop = np.random.rand(x_cp.shape[0],
                                  x_cp.shape[1]) < rate
  x_cp[pix_to_drop] = FLAGS.zero_bound
  return x_cp


def fill_feed_dict_ae(data_set, input_pl, target_pl, noise=None):
    input_feed, target_feed = data_set.next_batch(FLAGS.batch_size)
    if noise:
      input_feed = _add_noise(input_feed, noise)
    feed_dict = {
        input_pl: input_feed,
        target_pl: target_feed
    }
    return feed_dict


def fill_feed_dict(data_set, images_pl, labels_pl, noise=False):
  """Fills the feed_dict for training the given step.
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }
  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size ` examples.
  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)
  if noise:
      images_feed = _add_noise(images_feed, FLAGS.drop_out_rate)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict
