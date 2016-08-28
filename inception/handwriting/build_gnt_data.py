# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts gnt data to TFRecords file format with Example protos.

Running this script using 16 threads may take around ~2.5 hours on a HP Z420.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading
import io
from PIL import Image
import cPickle


import numpy as np
import tensorflow as tf

from read_gnt import *

tf.app.flags.DEFINE_string('train_directory', '/tmp/',
                           'Training data directory')
tf.app.flags.DEFINE_string('validation_directory', '/tmp/',
                           'Validation data directory')
tf.app.flags.DEFINE_string('output_directory', '/tmp/',
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 239,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 60,
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 8,
                            'Number of threads to preprocess the images.')


FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(image_buffer, label, height, width, human):
  """Build an Example proto for an example.
  """
  im = image_buffer.astype(np.uint8)
  im = Image.fromarray(im)
  bytes = io.BytesIO()
  im.save(bytes, 'jpeg')
  image_bytes = bytes.getvalue()

  example = tf.train.Example(features=tf.train.Features(feature={
    #'height': _int64_feature(height),
    #'width': _int64_feature(width),
    'label': _int64_feature(label),
    'text': _bytes_feature(human),
    'encoded': _bytes_feature(image_bytes)}))
  return example


def _process_image_files_batch(thread_index, ranges, name, filenames,
                               num_shards):
  """Processes and saves list of images as TFRecord in 1 thread.
  """
  num_threads = len(ranges)
  myrange = ranges[thread_index]
  num_shards_per_thread = myrange[1] - myrange[0]

  counter = 0
  for s in xrange(myrange[0], myrange[1]):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = s
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    filename = filenames[s]
    sys.stdout.write(filename)

    for text, label, image_buffer, width, height in GntFiles().load_file(filename):
      example = _convert_to_example(image_buffer, label, height, width, text)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

    print('%s [thread %d]: Processed %d images in file %s.' %
          (datetime.now(), thread_index, shard_counter, filename))
    sys.stdout.flush()

  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_shards_per_thread))
  sys.stdout.flush()


def _process_image_files(name, filenames, num_shards):
  """Process and save list of images as TFRecord of Example protos.
  """
  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
  ranges = []
  threads = []
  for i in xrange(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i+1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  threads = []
  for thread_index in xrange(len(ranges)):
    args = (thread_index, ranges, name, filenames,
            num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames)))
  sys.stdout.flush()


def _find_image_files(data_dir):
  """Build a list of all images files and labels in the data set.
  """
  filenames = []

  # Construct the list of JPEG files and labels.
  jpeg_file_path = '%s/*.gnt' % (data_dir)
  matching_files = tf.gfile.Glob(jpeg_file_path)
  filenames.extend(matching_files)

  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  shuffled_index = range(len(filenames))
  random.seed(12345)
  random.shuffle(shuffled_index)

  filenames = [filenames[i] for i in shuffled_index]

  print('Found %d JPEG files inside %s.' %
        (len(filenames), data_dir))
  return filenames


def _process_dataset(name, directory, num_shards):
  """Process a complete data set and save it as a TFRecord.
  """
  filenames = _find_image_files(directory)
  _process_image_files(name, filenames, num_shards)

def main(unused_argv):
  print('Saving results to %s' % FLAGS.output_directory)

  # Run it!
  _process_dataset('validation', FLAGS.validation_directory,
                   FLAGS.validation_shards)
  _process_dataset('train', FLAGS.train_directory, FLAGS.train_shards)

  pickle_label(os.path.join(FLAGS.output_directory, 'labels_output.pkl'))

if __name__ == '__main__':
  tf.app.run()
