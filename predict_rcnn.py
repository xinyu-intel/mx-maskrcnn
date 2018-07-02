import numpy as np
import mxnet as mx
import cv2
import time
from rcnn.io.image import resize, transform
from rcnn.symbol import *
from rcnn.config import config, default, generate_config
from rcnn.core.tester import Predictor
import pyximport
from mxboard import SummaryWriter

network = 'resnet_fpn'
net = eval('get_' + network + '_mask_test')(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)

# some config info
dev_id = 0
DATA_NAMES = ['data', 'im_info']
LABEL_NAMES = None
DATA_SHAPES = [('data', (1,3,300,300)), ('im_info', (1,3))]
LABEL_SHAPES = None
PIXEL_MEANS = np.array([103.939, 116.779, 123.68])

#create predictor
predictor = Predictor(net, 
                    DATA_NAMES, LABEL_NAMES,
                    context = mx.cpu(), # mx.gpu(dev_id),
                    provide_data = DATA_SHAPES,
                    provide_label = LABEL_SHAPES)

#process input data from random and create image info
def mem_data(h=500, w=353):
    im = np.random.randint(low=0, high=256, size=(3,h,w))
    im = im.astype('uint8')
    im_array, im_scale = resize(im, 3, 300)
    im_tensor = np.zeros((1,3, h, w))
    im_tensor[0,:,:,:] = im_array[:,:,:] - PIXEL_MEANS[2]
    im_array = im_tensor
    im_info = np.array([[im_array.shape[2], im_array.shape[3], im_scale]], dtype = np.float32)
    data = [mx.nd.array(im_array), mx.nd.array(im_info)]
    data_shapes = [('data', im_array.shape), ('im_info', im_info.shape)]
    data_batch = mx.io.DataBatch(data=data, label=None, 
                                 provide_data=data_shapes, 
                                 provide_label=None)
    return data_batch
    
def benchmark(loop=1):
    mem_datas = []
    for i in range(loop + 0):
        data_batch = mem_data(300,300)
        mem_datas.append(data_batch)
    mx.profiler.set_config(profile_all=True, filename='profile_output_native.json')
    mx.profiler.set_state('run')
    tic = time.time()
    for i in range(loop + 0):
        if i==5:
            tic = time.time()
        # Predictor need add some code to sync the outputs in GPU
        # add in rcnn/core/tester.py:line 48 with function predict
        # 47 self._mod.forward(data_batch)
        # 48 for output in self._mod.get_outputs():
        # 49     output.wait_to_read()
        # 50 return xx
        predictor.predict(mem_datas[i])
    mx.profiler.set_state('stop')
    cost = (time.time() - tic) / loop
    print("Average cost time is {}, img/sec is {}".format(cost, 1/cost))
    
if __name__ == '__main__':
    benchmark()
