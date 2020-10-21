import numpy as np
import h5py


def extract_images(filename, imageName):
  """Extract the images into a 3D uint8 numpy array [index, y, x, depth]."""
  print(filename)
  print(imageName)
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

  def __init__(self, images, flip90=False):
    """Construct a DataSet"""

    self._num_examples = images.shape[0]

    if flip90:
      res = self.rotate90(images)
      images = np.concatenate([images, res], axis=0)
    self._images = images

    self._epochs_completed = 0
    self._index_in_epoch = 0

  @staticmethod
  def rotate90(images):
    res = np.zeros(images.shape)
    for k in range(images.shape[0]):
      res[k,...] = np.rot90(images[k,...])
    return res

  @property
  def image_resolution(self):
    shape = self._images.shape
    return (shape[1], shape[2])
  
  @property
  def imagTru(self):
    return self._images

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  @property
  def data(self):
    return self._images

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

      self._images= self._images[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return np.expand_dims(self._images[start:end], axis=-1)

  def default_batch(self, batch_size, starting_index=0):
    return np.expand_dims(self._images[starting_index:(starting_index+batch_size)], axis=-1)


def read_data_sets(trainFileName, testFileName, vessels=False, flip90=False):
  class DataSets(object):
      pass
  data_sets = DataSets()

  TRAIN_SET = trainFileName
  TEST_SET  = testFileName

  if vessels:
    IMAGE_NAME = 'smallPhan'
  else:
    IMAGE_NAME = 'imagesTrue'

  print('Start loading test data')
  test_images = extract_images(TEST_SET, IMAGE_NAME)
  data_sets.test = DataSet(test_images, flip90=flip90)

  print('Start loading training data')
  train_images= extract_images(TRAIN_SET, IMAGE_NAME)
  data_sets.train = DataSet(train_images, flip90=flip90)

  return data_sets
