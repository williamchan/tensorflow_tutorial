"""
fully_connected_feed.py trains the built MNIST model against the downloaded dataset using a feed dictionary. It is written to be run from the command line

$ python fully_connected_feed.py

$ python3 fully_connected_feed.py --help
usage: fully_connected_feed.py [-h] [--learning_rate LEARNING_RATE]
                               [--max_steps MAX_STEPS] [--hidden1 HIDDEN1]
                               [--hidden2 HIDDEN2] [--batch_size BATCH_SIZE]
                               [--input_data_dir INPUT_DATA_DIR]
                               [--log_dir LOG_DIR] [--fake_data]

optional arguments:
  -h, --help            show this help message and exit
  --learning_rate LEARNING_RATE
                        Initial learning rate.
  --max_steps MAX_STEPS
                        Number of steps to run trainer.
  --hidden1 HIDDEN1     Number of units in hidden layer 1.
  --hidden2 HIDDEN2     Number of units in hidden layer 2.
  --batch_size BATCH_SIZE
                        Batch size. Must divide evenly into the dataset sizes.
  --input_data_dir INPUT_DATA_DIR
                        Directory to put the input data.
  --log_dir LOG_DIR     Directory to put the log data.
  --fake_data           If true, uses fake data for unit testing.

"""

# These 3 lines provides backward compatibility with older Python versions from Python 3 code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# six is a package that helps in writing code that is compatible with both Python 2 and Python 3.
from six.moves import xrange  # pylint: disable=redefined-builtin

import argparse
import os.path
import sys
import time
import math
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

# Basic model parameters as external flags.
FLAGS = None

def run_training():
  """Train MNIST for a number of steps."""

  # Get the sets of images and labels for training, validation, and test on MNIST.
  data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)

  # We start building the computation graph for the model here. Tell Tensorflow that
  # the model will be built into the default graph.  
  with tf.Graph().as_default():

    # Generate placeholders for the images and labels.
    images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size))
    
    # Debug mode - print out some stuffs
    print (images_placeholder.get_shape())
    print (labels_placeholder.get_shape())

    '''
    The Inference Engine
    - Build a Graph that computes predictions from the inference model.
    '''
    # Hidden 1
    with tf.name_scope('FC1'):
        # Created under the hidden1 scope, the unique name given to the weights variable would be "hidden1/weights".
        weights = tf.Variable(tf.truncated_normal([mnist.IMAGE_PIXELS, FLAGS.hidden1],
            stddev=1.0 / math.sqrt(float(mnist.IMAGE_PIXELS))), name='weights')
        # Likewise, the unique name given to the biases variable would be "hidden1/biases".
        biases = tf.Variable(tf.zeros([FLAGS.hidden1]), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images_placeholder, weights) + biases)

    # Hidden 2
    with tf.name_scope('FC2'):
        # "hidden2/weights"
        weights = tf.Variable(
            tf.truncated_normal([FLAGS.hidden1, FLAGS.hidden2],stddev=1.0 / math.sqrt(float(FLAGS.hidden1))),
            name='weights')
        # "hidden2/biases"
        biases = tf.Variable(tf.zeros([FLAGS.hidden2]), name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    # Linear
    with tf.name_scope('softmax_linear'):
        # "softmax_linear/weights"    
        weights = tf.Variable(tf.truncated_normal([FLAGS.hidden2, mnist.NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(FLAGS.hidden2))), name='weights')
        # "softmax_linear/biases" 
        biases = tf.Variable(tf.zeros([mnist.NUM_CLASSES]), name='biases')
        logits = tf.matmul(hidden2, weights) + biases
   
    '''
    Loss Function
    - Add to the Graph the Ops for loss calculation.
    '''
    with tf.name_scope('softmax'):
        labels = tf.to_int64(labels_placeholder) #typecasting in int64
    
        # This op produces 1-hot labels from the labels_placeholder and compare them against logits from the
        # inference engine
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')

        # This op averages the cross entropy values across the batch dimension (the first dimension) as the total loss.
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

    '''
    The Training Op
    - Add to the Graph the Ops that calculate and apply gradients.
    '''
    with tf.name_scope('adam_optimizer'):

        # tf.summary.scalar is an op for generating summary values into the events file when used with a 
        # tf.summary.FileWriter. In this case, it will emit the snapshot value of the loss every time the
        # summaries are written out.
        tf.summary.scalar('loss', loss)

        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(loss, global_step=global_step, name='minimize')

    '''
    The Evaluation Op
    - Add the Op to compare the logits to the labels during evaluation.
    '''
    with tf.name_scope('eval'):
        # For a classifier model, we can use the in_top_k Op. It returns a bool tensor with shape [batch_size] 
        # that is true for the examples where the label is in the top k (here k=1) of all logits for that example.
        correct = tf.nn.in_top_k(logits, labels, 1, name = 'top_k')
        # Return the number of true entries.
        eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32), name = 'reduce_sum')

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all() 
    
    # Add the variable initializer Op.
    init = tf.global_variables_initializer()    # Create a saver for writing training checkpoints.

    # Create a session for running Ops on the Graph.
    sess = tf.Session()
    
    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
    
    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)
    
    '''
    The Train Loop
    '''
    for step in xrange(FLAGS.max_steps):

        # Create the feed_dict for the placeholders filled with the next
        # `batch size` examples.
        images_feed, labels_feed = data_sets.train.next_batch(FLAGS.batch_size, FLAGS.fake_data)
        feed_dict = {
          images_placeholder: images_feed,
          labels_placeholder: labels_feed,
        }

        # Run one step of the model.  The return values are the activations
        # from the `train_op` (which is discarded) and the `loss` Op.  
        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)


        # Print an overview every 100 steps
        if step % 100 == 0:
            # Print status to stdout.
            print('Step %d: loss = %.2f' % (step, loss_value))

    summary_writer.close()
    
def main(_):
    
  # Deal with the Log file  
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)  # Delete everything in the log directory if it already exists
    
  tf.gfile.MakeDirs(FLAGS.log_dir)  # Create the directory if it does not exist already
  
  run_training()  # Start training the model


if __name__ == '__main__':

  # The argparse module makes it easy to write user-friendly command-line interfaces.
  # Running <python fully_connected_feed.py -h> provides useful help messages describing
  # the optional arguments outlined below.
  parser = argparse.ArgumentParser()  # Without a program name, ArgumentParser determine the command-line arguments from sys.argv
    
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='Initial learning rate.'
  )

  parser.add_argument(
      '--max_steps',
      type=int,
      default=10000,
      help='Number of steps to run trainer.'
  )
    
  parser.add_argument(
      '--hidden1',
      type=int,
      default=128,
      help='Number of units in hidden layer 1.'
  )

  parser.add_argument(
      '--hidden2',
      type=int,
      default=32,
      help='Number of units in hidden layer 2.'
  )
    
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )

  parser.add_argument(
      '--input_data_dir',
      type=str,
      default='/tmp/tensorflow/mnist/input_data',
      help='Directory to put the input data.'
  )
    
  parser.add_argument(
      '--log_dir',
      type=str,
      default='/home/lukeliem/TensorFlow/logs/fully_connected_feed',
      help='Directory to put the log data.'
  )

  parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )
  
  # Sometimes a script may only parse a few of the command-line arguments, passing the remaining arguments on to another 
  # script or program. parse_known_args() returns a two item tuple containing the populated namespace (into FLAG) and the
  # list of remaining argument strings.
  FLAGS, unparsed = parser.parse_known_args()

  # Runs the program with an optional 'main' function and 'argv' list.
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)  # sys.argv is the list of command line arguments passed to the Python script
