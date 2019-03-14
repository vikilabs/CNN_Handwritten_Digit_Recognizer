'''
    PyTorch Wrapper to work with MNIST Handwritten Digit Database

    Author:

        Vignesh Natarajan (a) Viki
        viki@vikilabs.org
'''


import torchvision
from torchvision import transforms
import torch
from torchvision import datasets
import os
import errno
from VikiLabs_Logger import *

log = logger()

def Download_MNIST_TrainingData(path):
    print(log._st+ "DOWNLOADING MNIST TRAINING DATA")
    t = transforms 
    tf = t.Compose([t.ToTensor(), t.Normalize((0.5,), (0.5,))])
    data_object = datasets.MNIST(path, train=True, download=True, transform=tf)
    print(log._ed+ "DOWNLOADING MNIST TRAINING DATA")
    return data_object


def Download_MNIST_TestData(path):
    print(log._st+ "DOWNLOADING MNIST TEST DATA")
    t = transforms 
    
    '''
        Convert Image from range [0, 1] to  range [-1 to 1]
        
        image = (image - n_mean)/n_std
    '''
    n_mean = 0.5
    n_std  = 0.5


    tf = t.Compose([t.ToTensor(), t.Normalize((n_mean,), (n_std,))])
    data_object  = datasets.MNIST(path, train=False, download=True, transform=tf)
    print(log._ed+ "DOWNLOADING MNIST TEST DATA")
    return data_object

def Load_MNIST_Data(data_object, batch_size):
    print(log._st+ "LOADING MNIST DATA")
    tud = torch.utils.data
    data = tud.DataLoader(data_object, batch_size=batch_size, shuffle=True)
    print(log._ed+ "LOADING MNIST DATA")
    return data 


def save_image(numpy_array, file_name):
    image_name = file_name + str(".png")
    tensor_array = torch.from_numpy(numpy_array)
    torchvision.utils.save_image(tensor_array, image_name)

def StoreDataAsImage(mnist_data, dfolder):
    
    try:
        os.mkdir(dfolder)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    file_base = "number"
    
    '''
    MNIST training data has 938 records. Each record in MNIST has the following
        1. images of shape [64, 1, 28, 28]  -> 64 handwritten digits
        2. labels for images of shape [64]  -> 64 label for the 64 handwritten digit images
    '''
    '''Full Download : 938'''
    #no_records_to_store = len(mnist_data)

    '''Only 64 Images Download'''
    no_records_to_store = 1

    #Iterate Over MNIST DATA
    for i, data in enumerate(mnist_data, 0):
        
        if(i >= no_records_to_store):
            break

        images, labels = data

        for j in range(len(images)):
            file_name = dfolder+str("/")+file_base+"_"+str(labels[j].item())+"_"+str(i)+"_"+str(j)              

            '''
            Pixel Values will be in range between -1 and 1
            '''
            normalized_image = images[j][0]
            n_mean = 0.5
            n_std = 0.5
            '''
            Pixel Values will be in range between 0 and 1
            '''
            denormalized_image = (normalized_image * n_std) + n_mean 
            image_numpy_array = denormalized_image.numpy()
            save_image(image_numpy_array, file_name)

'''
mnist_path = './data'
image_path = './images'
training_batch_size = 64
test_batch_size = 1000

training_object = Download_MNIST_TrainingData(mnist_path)
test_object     = Download_MNIST_TestData(mnist_path)

training_data   = Load_MNIST_Data( training_object, training_batch_size   )
test_data       = Load_MNIST_Data( test_object,     test_batch_size       )

StoreDataAsImage(training_data, image_path)
'''
