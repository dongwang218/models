import os.path
import pickle
import sys

# This is a placeholder for a Google-internal import.
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '../'))

import tensorflow as tf
from tensorflow.python.framework import tensor_shape, graph_util
from tensorflow.python.platform import gfile

import inception_model


tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/inception_train',
                           """Directory where to read training checkpoints.""")
tf.app.flags.DEFINE_string('export_dir', '/tmp/inception_export',
                           """Directory where to export inference model.""")
tf.app.flags.DEFINE_integer('image_size', 74,
                            """Needs to provide same value as in training.""")
FLAGS = tf.app.flags.FLAGS


NUM_CLASSES = 3755
NUM_TOP_CLASSES = 5

WORKING_DIR = os.path.dirname(os.path.realpath(__file__))


def export():

  with tf.Graph().as_default():
    # Build inference model.
    # Please refer to Tensorflow inception model for details.

    # Input transformation. between [-1, 1]
    batch_input = tf.placeholder(tf.float32,
                                 [None, 74, 74, 3],
                                 name='batch_input')

    # Run inference.
    logits, _ = inception_model.inference(batch_input, NUM_CLASSES + 0)

    predictions = tf.nn.softmax(logits, name = 'predictions')

    # Restore variables from training checkpoint.
    variable_averages = tf.train.ExponentialMovingAverage(
        inception_model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    with tf.Session() as sess:
      # Restore variables from training checkpoints.
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/imagenet_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('Successfully loaded model from %s at step=%s.' %
              (ckpt.model_checkpoint_path, global_step))
      else:
        print('No checkpoint file found at %s' % FLAGS.checkpoint_dir)
        return

      output_graph_def = graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), ['predictions'])
      with gfile.FastGFile(os.path.join(FLAGS.export_dir, 'output_graph.pb'), 'wb') as f:
        f.write(output_graph_def.SerializeToString())

def main(unused_argv=None):
  export()


if __name__ == '__main__':
  tf.app.run()
