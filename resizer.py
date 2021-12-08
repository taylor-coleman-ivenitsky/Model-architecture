import numpy as np
import tensorflow as tf
from PIL import Image
import random

def process():

  train_image = []
  train_labels = []
  test_image = []
  test_labels = []
  dict = {}
  date_list = []

  with open("all_dates.txt") as dates:

    #retrieve date from dates file for each image
    for line in dates:
      list = line.split(", ")
      id = list[0]
      date = list[1]
      date = date[0:4]
      date = int(date)

      #store the date in the dictionary with the image id as the key
      dict[id] = date

      #add the date to the dates list for each date
      date_list.append(date)

  #close the file
  dates.close()

  #for each corresponding date
  for id in dict:
    #read in the image and convert it to a numpy array
    image = "all_images/image" + id + ".png"
    img = Image.open(image)
    arr_img = np.array(img)
    arr_img = arr_img[...,:3]
    #x = np.shape(arr_img)[0]
    #y = np.shape(arr_img)[1]
    arr_img = tf.convert_to_tensor(arr_img)
    #arr_img = tf.reshape(arr_img, [x,y,3])

    #resize the image to the standard size for VGG
    res_image = tf.image.resize(arr_img, [224,224], method="lanczos3")

    #switch out the dates for the class labels in the dictionary
    date = dict[id]
    if date < 1800:
      label = 0
    elif 1800 <= date < 1900:
      label = 1
    elif 1900 <= date < 1920:
      label = 2
    elif 1920 <= date < 1940:
      label = 3
    elif 1940 <= date < 1960:
      label = 4
    else:
      label = 5

    #randomly place each image/date into either the train or test set
    rand = random.randint(1,5)
    if rand == 1:
      test_image.append(res_image)
      test_labels.append(label)
    else:
      train_image.append(res_image)
      train_labels.append(label)

  traini = np.array(train_image)
  print(traini.shape)
  testi = np.array(test_image)
  trainl = np.array(train_labels)
  testl = np.array(test_labels)

  #write the train and test data to numpy files
  np.save('train_img', traini)
  np.save('test_img', testi)
  np.save('train_lab', trainl)
  np.save('test_lab', testl)

  return train_image, train_labels, test_image, train_labels

process()

