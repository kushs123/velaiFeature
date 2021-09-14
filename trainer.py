# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from keras.models import load_model
from keras import backend as keras
from tqdm.notebook import tqdm
import os
import argparse

def concatenate(array1, array2):
  assert array1.shape[1] == array2.shape[1] and len(array1.shape) == len(array2.shape) == 2
  conc = np.zeros((len(array1)+len(array2), array1.shape[1]))

def show_img(img):
  plt.figure()
  plt.imshow((img[:,:,0]+1)*127.5, cmap ='gray')
  
def augment(X, y, nreq_imgs = -1, n_duplicates = 2, verbose = 1):
  print("input dataset size : {}".format(len(X)))
  aug_arr = []
  neg_arr = []
  for i in range(len(y)):
    if y[i] == 1:
      aug_arr.append(i)
    else:
      neg_arr.append(i)

  negative_img = np.zeros((len(neg_arr), 224, 224, 1))
  for i in range(len(neg_arr)):
    negative_img[i] = X[neg_arr[i]]

  positive_img = np.zeros((len(aug_arr), 224, 224, 1))
  for i in range(len(aug_arr)):
    positive_img[i] = X[aug_arr[i]]

  Scaled_positive_img = np.zeros((len(aug_arr), 224, 224, 1))
  for i in range(len(aug_arr)):
    scale_size = 250    
    temp = cv2.resize((X[aug_arr[i]][:,:,0]), (scale_size, scale_size))
    Scaled_positive_img[i] = temp[(scale_size-224)//2 + 1:scale_size - (scale_size-224)//2 + 1,(scale_size-224)//2 + 1:scale_size - (scale_size-224)//2 + 1, None]
    

  # np.random.seed = 22

  Scaled_positive_img = np.concatenate((Scaled_positive_img, positive_img), axis = 0)
  transform_img = np.zeros((Scaled_positive_img.shape[0]*n_duplicates, 224,224,1))
  
  for i in range(len(Scaled_positive_img)):
    image = Scaled_positive_img[i]
    # show_img(image)
    shift = int(0.2*image.shape[0])
    for j in range(n_duplicates):
      tx = np.random.uniform(low = -shift, high = +shift)
      ty = np.random.uniform(low = -shift, high = +shift)
      M = np.float32([[1, 0, tx],	[0, 1, ty]])
      shifted = (np.reshape(cv2.warpAffine((image+1)*127.5, M, (image.shape[1], image.shape[0]), borderMode = cv2.BORDER_CONSTANT, borderValue=0), (224,224,1))-127.5)/127.5
      transform_img[i*n_duplicates + j] = shifted

  if nreq_imgs == -1:

    X_final = np.array(np.concatenate((transform_img, negative_img), axis = 0))
    y_final = np.array([1.0 if i<len(transform_img) else 0.0 for i in range(len(X_final))])
    if verbose:
      print("positive_images used: {}".format(len(transform_img)))
      print("negative_images used: {}".format(len(negative_img)))
    return X_final, y_final

  else:
    rng = np.random.default_rng()
    transform_img_pos_trimmed = rng.choice(transform_img, size = min(len(transform_img), nreq_imgs//2), axis = 0, replace = False)
    neg_img_trimmed = rng.choice(negative_img, size = min(len(negative_img), nreq_imgs-len(transform_img_pos_trimmed)), axis = 0, replace = False)
    if verbose:
      print("positive_images used: {}".format(len(transform_img_pos_trimmed)))
      print("negative_images used: {}".format(len(neg_img_trimmed)))

    X_final = np.array(np.concatenate((transform_img_pos_trimmed, neg_img_trimmed), axis = 0))
    y_final = np.array([1.0 if i<len(transform_img_pos_trimmed) else 0.0 for i in range(len(X_final))])

    return X_final, y_final

def generate_model(args):
  data = np.load(args.image_file_path)
  X = data['X']
  y = data['y']
  print("X.shape = ", X.shape)
  print("y.shape = ", y.shape)

  label_mapping = {
      0 : 'Aortic enlargement',
      1 : 'Atelectasis',
      2 : 'Calcification',
      3 : 'Cardiomegaly',
      4 : 'Consolidation',
      5 : 'ILD',
      6 : 'Infiltration',
      7 : 'Lung Opacity',
      8 : 'Nodule',
      9 : 'Other_lesion',
      10 : 'Pleural_effusion',
      11 : 'Pleural_thickening',
      12 : 'Pneumothorax',
      13 : 'Pulmonary_fibrosis'
  }

  label_reverse_mapping = {}
  for key in label_mapping:
    label_reverse_mapping[label_mapping[key]] = key

  label_counts = np.sum(y, axis = 0)
  categories = ['Nodule', 'Pleural_thickening', 'Pleural_effusion']
  for lab in categories:
    print("{}_positive/negative: {}/{}".format(lab, int(label_counts[label_reverse_mapping[lab]]), len(y)-int(label_counts[label_reverse_mapping[lab]])))

  # 3-fold analysis

  num_folds = 3
  for cat in categories[0:1]:
      category_mean_scores = []
      for j in range(num_folds):    
        print("Category: {}".format(cat))
        cat_index = label_reverse_mapping[cat]
        y_cat = y[:, cat_index]
        scores = []
        split_count = 1
        print_res = args.print_res
        # if print_res:
        #   print("Split Count : {}".format(split_count))
        # split_count+=1

        X_train_data, X_test, y_train_data, y_test = train_test_split(X[:4394], y_cat[:4394], test_size = 0.36, shuffle = True)
        X_new_train_data = np.concatenate((X_train_data, X[4394:]))
        y_new_train_data = np.concatenate((y_train_data, y_cat[4394:]))
        X_train, X_val, y_train, y_val = train_test_split(X_new_train_data, y_new_train_data, test_size = 0.2, shuffle = True)

        # X_train_data, X_test, y_train_data, y_test = train_test_split(X, y_cat, test_size = 0.2, shuffle = True)
        # X_train, X_val, y_train, y_val = train_test_split(X_train_data, y_train_data, test_size = 0.2, shuffle = True)
        # X_train_aug, y_train_aug = augment(X_train, y_train, nreq_imgs = -1)
        

        # model = Sequential()
        # model.add(Conv2D(32, 3, strides = (2,2), activation = 'relu', input_shape = (224,224,1)))
        # model.add(MaxPooling2D((2,2)))
        # model.add(Conv2D(32, 3, strides = (2,2), activation = 'relu'))
        # model.add(Dropout(0.5))
        # model.add(Conv2D(32, 3, strides = (2,2), activation = 'relu'))
        # model.add(MaxPooling2D((2,2)))
        # model.add(Conv2D(32, 3, activation = 'relu'))
        # model.add(Flatten())
        # model.add(Dense(128, activation = 'relu'))
        # model.add(BatchNormalization())
        # model.add(Dense(1, activation = 'sigmoid'))
        # model.compile(optimizer=Adam(learning_rate = 2e-4), loss = 'binary_crossentropy', metrics = ['AUC'])

        base_model = ResNet50(include_top=False, weights = None, input_shape=(224, 224, 1))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(1, activation= 'sigmoid')(x)
        model = Model(inputs = base_model.input, outputs = predictions)
        model.compile(optimizer=Adam(learning_rate = 1e-4), loss = 'binary_crossentropy', metrics = ['AUC', 'accuracy'])

        callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        
        # model.fit(X_train_aug, y_train_aug, batch_size = batch_size[i], validation_data=(X_val, y_val), epochs = 100, verbose = 1, callbacks = [callback])
        model.fit(X_train, y_train, batch_size = args.batch_size, validation_data=(X_val, y_val), epochs = 100, verbose = 1, callbacks = [callback])
        model.save(os.path.join(args.output_location, cat + "_batch_size_" + str(args.batch_size) + "_fold_" + str(j)))
        y_pred = model.predict(X_test)
        y_pred_label = np.array([0 if x < 0.5 else 1 for x in y_pred])
        tn, fp, fn, tp = confusion_matrix(y_test.astype(int), y_pred_label).ravel()
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        Accuracy = (tp+tn)/(tp+tn+fp+fn)
        F1 = tp/(tp+0.5*(fn+fp))
        scores.append([precision, recall, Accuracy, F1])
        if print_res:
          print("precision = {}".format(precision))
          print("recall = {}".format(recall))
          print("Accuracy = {}".format(Accuracy))
          print("F1 = {}\n".format(F1))
      mean_scores = np.mean(np.array(scores), axis=0)
      category_mean_scores.append(mean_scores)
      print("Category: {}".format(cat))
      print("Mean precision = {}".format(mean_scores[0]))
      print("Mean recall = {}".format(mean_scores[1]))
      print("Mean Accuracy = {}".format(mean_scores[2]))
      print("Mean F1 = {}\n".format(mean_scores[3]))


if __name__=='__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--image_file_path', action='store', help='file in .npz format with images and corresponding labels')
  parser.add_argument('--output_location', action='store', help='Output model folder path')
  parser.add_argument('--print_res', action='store', type=bool, default=True, help='Whether to print results of each fold or not')
  parser.add_argument('--batch_size', action='store', type=int, default=8, help='batch_size for the given model')

  args=parser.parse_args()
  generate_model(args)

  