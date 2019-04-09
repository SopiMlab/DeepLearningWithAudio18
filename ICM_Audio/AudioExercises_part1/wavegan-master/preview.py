from __future__ import print_function
import pickle
import os
import time

import numpy as np
import tensorflow as tf
from six.moves import xrange

import loader
from wavegan import WaveGANGenerator, WaveGANDiscriminator
from functools import reduce

os.environ["CUDA_VISIBLE_DEVICES"]="0"



"""
usage example:
python preview.py train/piano/piano1 --noise_max 1 --noise_min -1
"""

"""
  Constants
"""
_FS = 16000
_WINDOW_LEN = 16384




import tempfile
import itertools as IT

def uniquify(path, sep = '_'):
    def name_sequence():
        count = IT.count()
        yield ''
        while True:
            yield '{s}{n:d}'.format(s = sep, n = next(count))
    orig = tempfile._name_sequence 
    with tempfile._once_lock:
        tempfile._name_sequence = name_sequence()
        path = os.path.normpath(path)
        dirname, basename = os.path.split(path)
        filename, ext = os.path.splitext(basename)
        fd, filename = tempfile.mkstemp(dir = dirname, prefix = filename, suffix = ext)
        tempfile._name_sequence = orig
    return filename

"""
  Generates a preview audio file every time a checkpoint is saved
"""
def preview(args):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from scipy.io.wavfile import write as wavwrite
  from scipy.signal import freqz

  ckpt_fp = args.checkpoint_file
  checkpointdirectory, checkpointname = os.path.split(ckpt_fp)

  preview_dir = os.path.join(checkpointdirectory, 'preview')
  if not os.path.isdir(preview_dir):
    os.makedirs(preview_dir)

  # Load graph
  infer_metagraph_fp = os.path.join(checkpointdirectory, 'infer', 'infer.meta')
  graph = tf.get_default_graph()
  saver = tf.train.import_meta_graph(infer_metagraph_fp)

  # Generate or restore z_i and z_o
  z_fp = os.path.join(preview_dir, 'z.pkl')
  if args.use_prev_noise and os.path.exists(z_fp):
    with open(z_fp, 'rb') as f:
      _zs = pickle.load(f)
  else:
    # Sample z
    samp_feeds = {}
    samp_feeds[graph.get_tensor_by_name('samp_z_n:0')] = args.preview_n
    samp_feeds[graph.get_tensor_by_name('samp_z_max:0')] = args.noise_max
    samp_feeds[graph.get_tensor_by_name('samp_z_min:0')] = args.noise_min
    samp_fetches = {}
    samp_fetches['zs'] = graph.get_tensor_by_name('samp_z:0')
    with tf.Session() as sess:
      _samp_fetches = sess.run(samp_fetches, samp_feeds)
    _zs = _samp_fetches['zs']

    # Save z
    with open(z_fp, 'wb') as f:
      pickle.dump(_zs, f)

  # Set up graph for generating preview images
  feeds = {}
  feeds[graph.get_tensor_by_name('z:0')] = _zs
  feeds[graph.get_tensor_by_name('flat_pad:0')] = _WINDOW_LEN // 2
  fetches = {}
  fetches['step'] = tf.train.get_or_create_global_step()
  fetches['G_z'] = graph.get_tensor_by_name('G_z:0')
  fetches['G_z_flat_int16'] = graph.get_tensor_by_name('G_z_flat_int16:0')

  if ckpt_fp != None:
    print('---------Preview: {}'.format(ckpt_fp))
    with tf.Session() as sess:
      saver.restore(sess, ckpt_fp)

      _fetches = sess.run(fetches, feeds)
    
    savename = '{}_{}_{}'.format(checkpointname,args.noise_min,args.noise_max)

    preview_fp = os.path.join(preview_dir, '{}.wav'.format(savename))
    preview_fp = uniquify(preview_fp)
    wavwrite(preview_fp, _FS, _fetches['G_z_flat_int16'])

    print('Done')

if __name__ == '__main__':
  import argparse
  import glob
  import sys

  parser = argparse.ArgumentParser()

  parser.add_argument('checkpoint_file', type=str,
      help='checkpoint_file')

  preview_args = parser.add_argument_group('Preview')
  preview_args.add_argument('--preview_n', type=int,
      help='Number of samples to preview')
  preview_args.add_argument('--noise_min', type=float,
      help='min value for the input noise')
  preview_args.add_argument('--noise_max', type=float,
      help='max value for the input noise')
  preview_args.add_argument('--use_prev_noise', type=bool,
      help='whether to use previously used noise')


  parser.set_defaults(
    preview_n=32,
    checkpoint_name = None,
    noise_min = -1.0,
    noise_max = 1.0,
    use_prev_noise = False)

  args = parser.parse_args()

  preview(args)
