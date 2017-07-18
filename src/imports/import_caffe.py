__author__ = 'joon'

caffe_root = 'caffe/'

import sys

sys.path.insert(0, caffe_root + 'python')
import caffe  # If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

sys.path.append(caffe_root + "examples/pycaffe")  # the tools file is in this folder
import tools  # this contains some tools that we need
import tools as caffetools  # this contains some tools that we need

sys.path.insert(0, 'lib')


def disp_net(net):
    print ('\n=====\nBLOBS\n=====\n')
    for layer_name, blob in net.blobs.iteritems():
        print layer_name + '\t' + str(blob.data.shape)

    print ('\n=====\nLAYERS\n=====\n')
    for layer_name, param in net.params.iteritems():
        if len(param) > 1:
            print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)
        else:
            print layer_name + '\t' + str(param[0].data.shape)


def set_preprocessor_without_net(data_shape, mean_image=None):
    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': data_shape})

    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    if mean_image is not None:
        transformer.set_mean('data', mean_image)  # subtract the dataset-mean value in each channel
    # transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    return transformer
