import numpy as np
import tensorflow     # only used to get rid of error on gradescope
from keras.datasets import mnist, fashion_mnist

def load_data(dataset):
    
    if dataset == "fashion_mnist":
        (x_train,y_train),(x_test,y_test)= fashion_mnist.load_data()
    else:
        (x_train,y_train),(x_test,y_test)=  mnist.load_data()

    x_train = x_train.reshape(len(x_train), -1)/255.0
    x_test  = x_test.reshape(len(x_test), -1)/255.0

    return x_train,y_train,x_test,y_test
