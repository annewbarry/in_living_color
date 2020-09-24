import tensorflow.keras.backend as kb
import tensorflow.math as tfmath
import os
import matplotlib.pyplot as plt

### This .py file mostly contains different loss functions I experimented with


def custom_loss_2k(y_actual, y_predicted):
    '''
    Takes actual and predicted values and returns a loss function value
    
    Input: y_actual, y_predicted (np arrays)
    Output: loss function value (float)
    
   '''

    cv = kb.abs(kb.mean((kb.square(y_actual - y_predicted))) - kb.mean(kb.square(y_predicted))/2000)
    return cv


def custom_loss_function(y_actual, y_predicted):
    '''
    Takes actual and predicted values and returns a loss function value
    
    Input: y_actual, y_predicted (np arrays)
    Output: loss function value (float)
    
   '''

    cv = kb.abs(kb.mean((kb.square(y_actual - y_predicted))) - kb.mean(kb.square(y_predicted))/2)
    return cv


def custom_loss_extreme(y_actual, y_predicted):
    '''
    Takes actual and predicted values and returns a loss function value
    
    Input: y_actual, y_predicted (np arrays)
    Output: loss function value (float)
    
   '''

    cv = kb.abs(kb.mean((kb.square(y_actual - y_predicted))/2) - kb.mean(kb.square(y_predicted))/2)
    return cv

def custom_loss_parabola(y_actual, y_predicted):
    '''
    Takes actual and predicted values and returns a loss function value
    
    Input: y_actual, y_predicted (np arrays)
    Output: loss function value (float)
    
   '''

    cv = -1*kb.mean((kb.square(y_actual - y_predicted))) + 1
    
    return cv     


def custom_loss_tens(y_actual, y_predicted):
    '''
    Takes actual and predicted values and returns a loss function value
    
    Input: y_actual, y_predicted (np arrays)
    Output: loss function value (float)
    
   '''

    cv = kb.abs(10*kb.mean((kb.square(y_actual - y_predicted))/2) - kb.mean(kb.square(y_predicted))/10)
    
    return cv  

def list_files(directory):
    '''
    Reads file names from a directory into a list
    
    Input: directory path (string)
    Output: List of strings
    
    '''
    
    files = []
    for filename in os.listdir(directory):
        if filename.endswith('jpg'):
            files.append(directory + filename)
    return files
