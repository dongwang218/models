import os, sys
import glob
import numpy, scipy.misc
import pickle
import gzip
import threading

# shared dict and its values
text2label = {}
text2label_lock = threading.Lock()
def get_label(text):
  """This rely on lock, not a good idea, should build this first"""
  with text2label_lock:
    label = text2label.get(text, None)
    found = True
    if not label:
      label = len(text2label)
      text2label[text] = label
      found = False
  return(found, label, text)

def get_num_labels():
  with text2label_lock:
    return(len(text2label))

def pickle_label(output_file):
  with text2label_lock:
    with open(output_file, 'wb') as output:
      pickle.dump(text2label, output)

class SingleGntImage(object):
  def __init__(self, f):
    self.f = f

  def read_gb_label(self):
    label_gb = self.f.read(2)

    # check garbage label
    if label_gb.encode('hex') is 'ff':
      return True, None
    else:
      label_utf8 = label_gb.decode('gb18030').encode('utf-8')
      return(get_label(label_utf8))

  def read_special_hex(self, length):
    num_hex_str = ""

    # switch the order of bits
    for i in range(length):
      hex_2b = self.f.read(1)
      num_hex_str = hex_2b + num_hex_str

    return int(num_hex_str.encode('hex'), 16)

  def read_single_image(self, image_size,
      image_margin, blur_type):

    # try to read next single image
    try:
      self.next_length = self.read_special_hex(4)
    except ValueError:
      return None, None, None, None, None, True

    # read the chinese utf-8 label
    self.is_garbage, self.label, self.text = self.read_gb_label()

    # read image width and height and do assert
    self.width = self.read_special_hex(2)
    self.height = self.read_special_hex(2)
    assert self.next_length == self.width * self.height + 10

    # read image matrix
    image_matrix_list = []
    for i in range(self.height):
      row = []
      for j in range(self.width):
        row.append(self.read_special_hex(1))

      image_matrix_list.append(row)

    # convert to ndarray with size of 40 * 40 and add margin of 4
    # (in default)
    self.image_matrix_numpy = scipy.misc.imresize(
      numpy.array(image_matrix_list),
      size=(image_size - 2 * image_margin,
        image_size - 2 * image_margin))
    self.image_matrix_numpy = numpy.lib.pad(self.image_matrix_numpy,
      image_margin, self.padwithones)

    # blur
    if blur_type == "gaussian":
      pass
    elif blur_type == "bi-value":
      pass
    elif blur_type == "bi-plus-gaussian":
      pass

    return self.text, self.label, self.image_matrix_numpy, \
      self.width, self.height, False

  def padwithones(self, vector, pad_width, iaxis, kwargs):
    # zero-one value
    max_value = 255

    vector[:pad_width[0]] =  max_value
    vector[-pad_width[1]:] = max_value
    return vector

class GntFiles(object):
  def __init__(self):
    pass

  def load_file(self, file_name, image_size=74,
      image_margin=2, blur_type="none"):

    #open all gnt files
    with open(file_name, 'rb') as f:
      end_of_image = False
      count_single = 0

      while not end_of_image:
        count_single += 1
        this_single_image = SingleGntImage(f)

        # get the pixel matrix of a single image
        text, label, pixel_matrix, width, height, end_of_image = \
          this_single_image.read_single_image(image_size,
                                              image_margin, blur_type)

        # load matrix ato 1d feature to array
        if not end_of_image:
          yield((text, label, pixel_matrix, width, height))

    print ("Finish file %s with %i chars. unique Char count %i") % \
      (file_name, count_single, get_num_labels())
