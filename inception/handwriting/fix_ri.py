
# one time script to assign ri to label 0, because conflicts and there is no text with label 0.
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

from build_gnt_data import _int64_feature, _bytes_feature

ri = u'\u65e5'

def local_convert_to_example(image_bytes, label, human):
  """Build an Example proto for an example.
  """
  example = tf.train.Example(features=tf.train.Features(feature={
    #'height': _int64_feature(height),
    #'width': _int64_feature(width),
    'label': _int64_feature(label),
    'text': _bytes_feature(human),
    'encoded': _bytes_feature(image_bytes)}))
  return example

count = 0
for file in sys.argv[1:]:
  dir = os.path.dirname(file)
  filename = os.path.basename(file)
  newfile = os.path.join(dir+'_fix', filename)
  writer = tf.python_io.TFRecordWriter(newfile)
  for serialized_example in tf.python_io.tf_record_iterator(file):
    count += 1

    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    label = example.features.feature['label'].int64_list.value[0]

    text_bytes = example.features.feature['text'].bytes_list.value[0].decode('utf-8')
    if text_bytes == ri:
      count += 1
      print('fix: label %d to 0 for %s' % (label, ri.encode('utf-8') ))
      image_bytes = example.features.feature['encoded'].bytes_list.value[0]
      text = text_bytes.encode('utf-8')
      example = local_convert_to_example(image_bytes, 0, text)
      serialized_example = example.SerializeToString()

    writer.write(serialized_example)
  writer.close()
  print('fixed %s' % newfile)

print('total fix %s' % count)
