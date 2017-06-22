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
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

# Basic model parameters as external flags.
FLAGS = None

def run_training():
  """Train MNIST for a number of steps."""
  # Get the sets of images and labels for training, validation, and
  # test on MNIST.

  # Tell TensorFlow that the model will be built into the default Graph.
    # Generate placeholders for the images and labels.

    # Build a Graph that computes predictions from the inference model.
 
    # Add to the Graph the Ops for loss calculation.

    # Add to the Graph the Ops that calculate and apply gradients.

    # Add the Op to compare the logits to the labels during evaluation.

    # Build the summary Tensor based on the TF collection of Summaries.
 
    # Add the variable initializer Op.
    
    # Create a saver for writing training checkpoints.

    # Create a session for running Ops on the Graph.

    # Instantiate a SummaryWriter to output summaries and the Graph.

    # And then after everything is built:

    # Run the Op to initialize the variables.

    # Start the training loop.

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
 
      # Write the summaries and print an overview fairly often.
         # Print status to stdout.
         # Update the events file.
 
      # Save a checkpoint and evaluate the model periodically.
        # Evaluate against the training set.
        # Evaluate against the validation set.
        # Evaluate against the test set.
 
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
      default='./logs/fully_connected_feed',
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
