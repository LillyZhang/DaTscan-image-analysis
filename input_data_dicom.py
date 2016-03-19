#This script is a modification of the input_dat.py file in the TensorFlow MNIST tutorial. 
#The original script is available at https://github.com/tensorflow/tensorflow.git
#Volumetric DaTscan SPECT is a neuroimaging study used to evalute for Parkinson's disease. Each DaTscan contains 91 slices and each slice contains 91 x 109 pixels encoded with 16-bit greyscale. A single slice at the level of the basal ganglia containing the most intense radiotracer activity was selected for TensorFlow input. A total of 1513 axial images were randomized and split into three parts: 1189 images for the training set, 108 images for the validation set, and 216 images for the test set. """


"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path
import tensorflow.python.platform
import numpy
import string
import dicom
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import png
import getopt, sys


def extract_images(directory, targetlist):
  """Extract the images into a 4D uint16 numpy array [index, y, x, depth]."""
  print('Extracting', directory)

  filelist = targetlist # targetlist is the list of DICOM images submitted for analysis
  filelist.sort()
  num_images = len(filelist)  
  counter = 0

  selectedSlice = 41   # Select any particular slice in the 3D DICOM volume for analysis

  filenames = []
  data = numpy.empty((num_images, 109, 91, 1))  # Each DaTsan image contains 109 x 91 pixels  
  visual = numpy.zeros((91, 109))

  for filename in filelist:
    filenames.append(filename)
    image = dicom.read_file(os.path.join(directory, filename))  #filename is two directories down from where this is running
    rows = image.Rows
    assert rows == 109
    cols = image.Columns
    assert cols == 91

    for x in range(0, 109):
      for y in range(0, 91):
        data[counter][x][y] = image.pixel_array[selectedSlice-1][x][y]     
        visual[y][x] = image.pixel_array[selectedSlice-1][x][y]
    counter = counter + 1
  
  # Visually check the numpy array by writing it to a png file
  f = open('ramp.png', 'wb')  
  w = png.Writer(109, 91, greyscale=True, bitdepth=16)   
  w.write(f, visual)
  f.close()

  return data, filenames
  


def dense_to_one_hot(labels_dense, num_classes = 2):
  #Convert class labels from scalars to one-hot vectors.
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot
     

def extract_labels(txtfile, filename_order, one_hot=False):
  """Extract the labels for normal subjects, Parkinson's disease patients, or subjects with Scans Without Evidence of Dopamine Deficiency (SWEDD) into a 1D uint8 numpy array [index]."""
  """ Normal and SWEDD subjects are grouped into label "0". Parkinson's subjects are grouped into label "1" """
  print('Extracting', txtfile)

  num_labels = len(filename_order)
  labelarray = numpy.empty(num_labels,dtype=numpy.int32)

  
  patient_id_to_label = {} 
  with open(txtfile, "r") as f: 
    for line in f:
      line = line.strip()
      patient_id, group, label = line.split(",")
      label = int(label)
      assert (group == "Control" and label == 0) or (group == "PD" and label == 1) or (group == "SWEDD" and label == 0)
      patient_id_to_label[patient_id] = label

  for i in range (len(filename_order)):
    filename = filename_order[i]
    row = filename.split("_")
    patient_id_row = row[1]
    assert (len(row)>=2)
    assert (row[0] == "PPMI")
    assert (len(patient_id_row) == 4)
    assert (patient_id_to_label.has_key(patient_id_row))
    labelarray[i] = patient_id_to_label[patient_id_row]


  if one_hot:
    return dense_to_one_hot(labelarray)

  print ('labelarray shape is ', labelarray.shape)  
  return labelarray
 

class DataSet(object):

  def __init__(self, images, labels, fake_data=False, one_hot=False,
               dtype=tf.float32):
    """Construct a DataSet.

    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = tf.as_dtype(dtype).base_dtype
    if dtype not in (tf.uint8, tf.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      assert images.shape[3] == 1
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2])
      
      if dtype == tf.float32:
        print ('DTYPE is ', dtype)
        # Convert from [0, 65536] -> [0.0, 1.0] as the normalization step for datscan int16 datatype (mnist is int8)
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 65535.0)#255.0)
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

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]




def read_command_line():
  trainlist=[]
  testlist=[]
  try: 
    opts, args = getopt.getopt(sys.argv[1:], "", ["train=", "test="])
  except getopt.GetoptError as err:
    # print help information and exit:
    print (str(err)) # will print something like "option -a not recognized"
    usage()
    sys.exit(2)
  for o, a in opts:  #opts is a list of pairs
    if o == "--train":
      trainlist = a.split(",")
    elif o =="--test":
      testlist = a.split(",")
    else:
      assert False, "unhandled option"
  
  assert len(trainlist)!=0
  assert len(testlist)!=0
  return trainlist, testlist




def read_data_sets(train_dir, fake_data=False, one_hot=False, dtype=tf.float32):
  class DataSets(object):
    pass
  data_sets = DataSets()

  if fake_data:
    def fake():
      return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)
    data_sets.train = fake()
    data_sets.validation = fake()
    data_sets.test = fake()
    return data_sets

  VALIDATION_SIZE = 108
    
  trainlist, testlist = read_command_line()

  train_images, train_images_filenames = extract_images("data/image_directory", trainlist)

  train_labels = extract_labels("data/label.csv", train_images_filenames, one_hot=one_hot)

  test_images, test_images_filenames = extract_images("data/image_directory", testlist)

  test_labels = extract_labels("data/label.csv", test_images_filenames, one_hot=one_hot)

  validation_images = train_images[:VALIDATION_SIZE]
  validation_labels = train_labels[:VALIDATION_SIZE]
  train_images = train_images[VALIDATION_SIZE:]
  train_labels = train_labels[VALIDATION_SIZE:]

  data_sets.train = DataSet(train_images, train_labels, dtype=dtype)
  data_sets.validation = DataSet(validation_images, validation_labels,
                                 dtype=dtype)
  data_sets.test = DataSet(test_images, test_labels, dtype=dtype)

  return data_sets
