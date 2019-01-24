import math, pickle

import numpy as np


def save(dataset, file):
    
    with open(file, 'wb') as fo:
        
        pickle.dump(dataset, fo)


def unpickle(file):
    
    with open(file, 'rb') as fo:
        
        dict = pickle.load(fo, encoding='bytes')
    
    return dict


def batch_run(function, images, batch_size=5000):
    '''
    function   : lambda function taking images with shape [N,H,W,C] as input
    images     : tensor of shape [N,H,W,C]
    batch_size : batch size
    '''
    
    res = []
    
    for i in range(math.ceil(len(images) / batch_size)):
        
        res.append(function(images[i*batch_size:(i+1)*batch_size]))
    
    return np.concatenate(res, axis=0)


def preprocess(attributions, q1, q2, use_abs=False):
    
    if use_abs:
        attributions = np.abs(attributions)
    
    attributions = np.sum(attributions, axis=-1)
    
    a_min = np.percentile(attributions, q1, axis=(1,2), keepdims=True)
    a_max = np.percentile(attributions, q2, axis=(1,2), keepdims=True)
    
    pos = np.tile(a_min > 0, [1,attributions.shape[1],attributions.shape[2]])
    ind = np.where(attributions < a_min)
    
    attributions = np.clip(attributions, a_min, a_max)
    attributions[ind] = (1 - pos[ind]) * attributions[ind]
    
    return attributions


def pixel_range(img):
    vmin, vmax = np.min(img), np.max(img)

    if vmin * vmax >= 0:
        
        v = np.maximum(np.abs(vmin), np.abs(vmax))
        
        return [-v, v], 'bwr'
    
    else:

        if -vmin > vmax:
            vmax = -vmin
        else:
            vmin = -vmax

        return [vmin, vmax], 'bwr'


def scale(x):
    
    return x / 127.5 - 1.0