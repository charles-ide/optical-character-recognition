'''
(c) 2023 Charles Ide
This is a project designed to explore optical character recognition with TensorFlow. This is the main file used to execute the program - all other functionality
should be refactored away to modules during development.
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
 
learn = tf.contrib.learn
 
tf.logging.set_verbosity(tf.logging.ERROR)

mnist = learn.datasets.load_dataset('mnist')
data = mnist.train.images
labels = np.asarray(mnist.train.labels, dtype=np.int32)
test_data = mnist.test.images
test_labels = np.asarray(mnist.test.labels, dtype=np.int32)