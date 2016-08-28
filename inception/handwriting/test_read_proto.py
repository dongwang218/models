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
"""Read and preprocess image data.
usage: python handwriting/test_read_proto.py label_output.pkl `ls ~/workspace/data/records/validation*`
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys
import numpy as np
from PIL import Image
from cStringIO import StringIO
import pickle

num_labels = 3755

labels = {}
count = 0
for file in sys.argv[2:]:
  for serialized_example in tf.python_io.tf_record_iterator(file):
    count += 1
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    #height = example.features.feature['height'].int64_list.value[0]
    #width = example.features.feature['width'].int64_list.value[0]
    label = example.features.feature['label'].int64_list.value[0]

    text_bytes = example.features.feature['text'].bytes_list.value[0].decode('utf-8')
    #print(text_bytes)
    labels[text_bytes] = label
    if count == 1:
      buffer = example.features.feature['encoded'].bytes_list.value[0] #jpeg

      file_jpgdata = StringIO(buffer)
      dt = Image.open(file_jpgdata)
      binary = np.array(dt) < 200
      for h in range(74):
        for w in range(74):
          if binary[h, w]:
            sys.stdout.write('.')
          else:
            sys.stdout.write(' ')
        sys.stdout.write('\n')
    if len(labels) == num_labels:
      print('total %s' % count)
      print('num labels found %d' % len(labels))

      with open(sys.argv[1], 'wb') as output:
        pickle.dump(labels, output)

      sys.exit(0)

print('total %s' % count)
print('num labels found %d' % len(labels))
