import numpy as np
import h5py


def extract_images(filename, imageName):
  """Extract the images into a 3D uint8 numpy array [index, y, x, depth]."""
  fData = h5py.File(filename,'r')
  inData = fData.get(imageName)  
      
  
  num_images = inData.shape[0]
  rows = inData.shape[1]
  cols = inData.shape[2]
  print(num_images, rows, cols)
  data = np.array(inData)
    
  data = data.reshape(num_images, rows, cols)
  data = np.transpose(data,(0,2,1))
  return data



class DataSet(object):

  def __init__(self, dataApr,dataTru,imagTru):
    """Construct a DataSet"""

    self._num_examples = dataApr.shape[0]


    self._dataApr = dataApr
    self._dataTru = dataTru
    self._imagTru = imagTru
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def image_resolution(self):
    shape = self._imagTru.shape
    return (shape[1], shape[2])

  @property
  def y_resolution(self):
    shape = self._dataTru.shape
    return (shape[1], shape[2])

  @property
  def dataApr(self):
    return self._dataApr

  @property
  def dataTru(self):
    return self._dataTru
  
  @property
  def imagTru(self):
    return self._imagTru

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._dataApr = self._dataApr[perm]
      self._dataTru = self._dataTru[perm]
      self._imagTru = self._imagTru[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return np.exp(self._dataApr[start:end], axis=-1), np.expand_dims(self._dataTru[start:end], axis=-1), \
           np.expand_dims(self._imagTru[start:end], axis=-1)


def read_data_sets(trainFileName, testFileName):
  class DataSets(object):
    pass
  data_sets = DataSets()

  TRAIN_SET = trainFileName
  TEST_SET  = testFileName
  DATAA_NAME = 'dataApprox'
  DATAT_NAME = 'dataTrue'
  IMAGE_NAME = 'imagesTrue'
  
  print('Start loading test data')
  test_dataApr = extract_images(TEST_SET,DATAA_NAME)
  test_dataTru   = extract_images(TEST_SET,DATAT_NAME)
  test_imagTru   = extract_images(TEST_SET,IMAGE_NAME)
  data_sets.test = DataSet(test_dataApr,test_dataTru,test_imagTru)

  print('Start loading training data')
  train_dataApr = extract_images(TRAIN_SET,DATAA_NAME)
  train_dataTru   = extract_images(TRAIN_SET,DATAT_NAME)
  train_imagTru   = extract_images(TRAIN_SET,IMAGE_NAME)
  data_sets.train = DataSet(train_dataApr,train_dataTru,train_imagTru)

  return data_sets