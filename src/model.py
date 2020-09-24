import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.compat.v1.image import resize_nearest_neighbor, resize
from tensorflow.keras.backend import resize_images
from skimage import io, color
import cv2
import os
from tensorflow.keras.layers import Conv2D, UpSampling2D, InputLayer
from skimage.color import lab2rgb, rgb2gray
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras.backend as kb
import tensorflow.math as tfmath

class MyModel():
    def __init__(self):
        self.model = None
        self.target_list = []
        
    def process_image(self, path_to_image):
        '''
        Takes an image from the directory, converts to lab space and splits into L layer (feature) and ab layer (target
        Input: image path (str)
        Output: feature (256, 256, 1 array), target(256, 256, 2 array)
        '''
        
        rgb = io.imread(path_to_image)
        rgb = cv2.resize(rgb, (256,256), interpolation=cv2.INTER_CUBIC)*1.0/255
        lab = color.rgb2lab(rgb)
        lab = img_to_array(lab)
#        else:
#             rgb = color.gray2rbg(io.imread(path_to_image, as_gray = True))
#             rgb = cv2.resize(rgb, (256,256), interpolation=cv2.INTER_CUBIC)*1.0/255
#             lab = color.rgb2lab(rgb)
#             lab = img_to_array(lab)
            
        feature = lab[:, :, 0]
        feature = feature.reshape(256,256,1)
        #print(feature)
        target = lab[:, :, 1:]/128
        return feature, target
    
    def image_generator(self, directory, batch_size = 10, return_x = False):
        '''
        creates list of feature and target images from directory
        
        Input: directory (str), batch_size (int)
        Output: yields list of feature and target images
        '''
        files = []
        for filename in os.listdir(directory)[:2000]:
            if filename.endswith('jpg'):
                files.append(directory + filename)        
        while True:
        # Select files (paths/indices) for the batch
            batch_paths  = np.random.choice(a = files, size = batch_size)
            batch_input  = []
            batch_output = [] 
          
          # Read in each input, perform preprocessing
            for input_path in batch_paths:
                feature, target = self.process_image(input_path)
                #print(f'Batch target is: {np.min(target)}, {np.max(target)}')
                batch_input += [feature] 
                batch_output += [target]
          # Return a tuple of (input, output) to feed the network
            batch_x = np.array( batch_input )
            batch_y = np.array( batch_output )
            yield(batch_x, batch_y)
            
    
    def make_model(self, loss, directory, optimizer = 'rmsprop', metrics = ['accuracy'], steps_per_epoch = 1, epochs = 1):
        model = Sequential()
        model.add(InputLayer(input_shape=(256, 256, 1)))
        #model.add(Conv2D(8, (3, 3), activation='tanh', padding='same', strides=2))
        model.add(Conv2D(128, (3, 3), activation='tanh', padding='same', strides=2))
        model.add(Conv2D(16, (3, 3), activation='tanh', padding='same'))
        model.add(Conv2D(16, (3, 3), activation='tanh', padding='same', strides=2))
        model.add(Conv2D(32, (3, 3), activation='tanh', padding='same', strides=2))
        model.add(Conv2D(32, (3, 3), activation='tanh', padding='same'))
        model.add(Conv2D(32, (3, 3), activation='tanh', padding='same', strides=2))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(32, (3, 3), activation='tanh', padding='same'))
        model.add(Conv2D(32, (3, 3), activation='tanh', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(16, (3, 3), activation='tanh', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(2, (3,3), activation='tanh', padding='same'))
        model.add(UpSampling2D((2, 2)))
### added this:
    #model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='tanh'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(2,
                  activation='tanh'))
        
        def custom_loss_function(y_actual, y_predicted):
            custom_loss_value = kb.abs(kb.mean((kb.square(y_actual - y_predicted)))- 
kb.mean(kb.square(y_predicted)/10))
            return custom_loss_value
        
        model.compile(optimizer,
              loss,
              metrics)
        #tensorboard = TensorBoard(log_dir='./logs', histogram_freq=2, batch_size=1500, write_graph=True, write_grads=True, write_images=True)
        model.fit(self.image_generator(directory), steps_per_epoch = steps_per_epoch,epochs = epochs)#, callbacks = [tensorboard])
        self.model = model
    
    def predict_process(self, path_to_image):
        X, _ = self.process_image(path_to_image)
        #print(X.shape)
        #print(f'X before reshape min is {np.min(X)} and max is {np.max(X)}')
        X = X.reshape(1, 256,256,1)
        #print(f'Xreshape {np.min(X[0, :, :, 0])}, {np.max(X[0, :, :, 0])}')
        output = self.model.predict(X)
        #print(f'Output before mult by 128: {np.min(output[:, :, :, 0])}, {np.max(output[:, :, :, 0])}')
        output *= 128
        #print(f'Output before mult by 128: {np.min(output[:, :, :, 1])}, {np.max(output[:, :, :, 1])}')
        img = []
        for i in range(len(output)):
            cur = np.zeros((256, 256, 3))
            cur[:,:,0] = X[i][:,:,0]
            cur[:,:,1:] = output[i]
            rgb_img = lab2rgb(cur)
            img.append(rgb_img)
#             print(f'(layer 1: {np.min(rgb_img[:, :, 0])}, {np.max(rgb_img[:, :, 0])}')
#             print(f'(layer 2: {np.min(rgb_img[:, :, 1])}, {np.max(rgb_img[:, :, 1])}')
#             print(f'(layer 3: {np.min(rgb_img[:, :, 2])}, {np.max(rgb_img[:, :, 2])}')
        return img
   
                
                
    
    
        
        
       