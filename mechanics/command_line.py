import argparse

# Basic model parameters as external flags.
FLAGS = None

def main():
  # Deal with the Log file  
  # TBD
  print("There is no code yet in this python file!!!")  # This is the main program!!!

# Test whether your script is being run directly or being imported by something. Only run the code below if the script is
# run directly!!!
if __name__ == '__main__':
    
  print("This script is being run directly")

  # The argparse module makes it easy to write user-friendly command-line interfaces.
  # Running <python fully_connected_feed.py -h> provides useful help messages describing
  # the optional arguments outlined below.
  parser = argparse.ArgumentParser()
    
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
      default='/tmp/tensorflow/mnist/logs/fully_connected_feed',
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
  main()

else:
  print("The script is being imported into another module")