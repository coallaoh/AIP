__author__ = 'joon'

import sys

sys.path.insert(0, 'src')
from imports.import_caffe import *
from caffe import layers as L
from caffe import params as P

sys.path.insert(0, 'lib')
from caffetools.basic_layers import get_learned_param, get_frozen_param, conv_relu, fc_relu, max_pool, conv, scale, \
    sigmoid, ptwisemult, softmask, upsample, bw2rgb, selector, select_by_masking, conv_bn_scale_relu, lrn

sys.path.insert(0, 'pynetbuilder/netbuilder')
from lego.base import BaseLegoFunction


def resnet_filler():
    weight_filler = dict(type='msra')
    bias_filler = dict(type='constant', value=0)
    conv_params = [weight_filler, bias_filler]
    return conv_params


def learn_layers_googlenet(control=dict()):
    if 'f_freeze' in control.keys():
        if control['f_freeze'] == 'notlast':
            ft_notlast = 0
        else:
            ft_notlast = 1
    else:
        ft_notlast = 1

    learn_this_layer = {}
    learn_this_layer['conv1_7x7_s2'] = ft_notlast
    learn_this_layer['conv2_3x3_reduce'] = ft_notlast
    learn_this_layer['conv2_3x3'] = ft_notlast

    learn_this_layer['inception_3a_1x1'] = ft_notlast
    learn_this_layer['inception_3a_3x3_reduce'] = ft_notlast
    learn_this_layer['inception_3a_3x3'] = ft_notlast
    learn_this_layer['inception_3a_5x5_reduce'] = ft_notlast
    learn_this_layer['inception_3a_5x5'] = ft_notlast
    learn_this_layer['inception_3a_pool_proj'] = ft_notlast

    learn_this_layer['inception_3b_1x1'] = ft_notlast
    learn_this_layer['inception_3b_3x3_reduce'] = ft_notlast
    learn_this_layer['inception_3b_3x3'] = ft_notlast
    learn_this_layer['inception_3b_5x5_reduce'] = ft_notlast
    learn_this_layer['inception_3b_5x5'] = ft_notlast
    learn_this_layer['inception_3b_pool_proj'] = ft_notlast

    learn_this_layer['inception_4a_1x1'] = ft_notlast
    learn_this_layer['inception_4a_3x3_reduce'] = ft_notlast
    learn_this_layer['inception_4a_3x3'] = ft_notlast
    learn_this_layer['inception_4a_5x5_reduce'] = ft_notlast
    learn_this_layer['inception_4a_5x5'] = ft_notlast
    learn_this_layer['inception_4a_pool_proj'] = ft_notlast

    learn_this_layer['inception_4b_1x1'] = ft_notlast
    learn_this_layer['inception_4b_3x3_reduce'] = ft_notlast
    learn_this_layer['inception_4b_3x3'] = ft_notlast
    learn_this_layer['inception_4b_5x5_reduce'] = ft_notlast
    learn_this_layer['inception_4b_5x5'] = ft_notlast
    learn_this_layer['inception_4b_pool_proj'] = ft_notlast

    learn_this_layer['inception_4c_1x1'] = ft_notlast
    learn_this_layer['inception_4c_3x3_reduce'] = ft_notlast
    learn_this_layer['inception_4c_3x3'] = ft_notlast
    learn_this_layer['inception_4c_5x5_reduce'] = ft_notlast
    learn_this_layer['inception_4c_5x5'] = ft_notlast
    learn_this_layer['inception_4c_pool_proj'] = ft_notlast

    learn_this_layer['inception_4d_1x1'] = ft_notlast
    learn_this_layer['inception_4d_3x3_reduce'] = ft_notlast
    learn_this_layer['inception_4d_3x3'] = ft_notlast
    learn_this_layer['inception_4d_5x5_reduce'] = ft_notlast
    learn_this_layer['inception_4d_5x5'] = ft_notlast
    learn_this_layer['inception_4d_pool_proj'] = ft_notlast

    learn_this_layer['inception_4e_1x1'] = ft_notlast
    learn_this_layer['inception_4e_3x3_reduce'] = ft_notlast
    learn_this_layer['inception_4e_3x3'] = ft_notlast
    learn_this_layer['inception_4e_5x5_reduce'] = ft_notlast
    learn_this_layer['inception_4e_5x5'] = ft_notlast
    learn_this_layer['inception_4e_pool_proj'] = ft_notlast

    learn_this_layer['inception_5a_1x1'] = ft_notlast
    learn_this_layer['inception_5a_3x3_reduce'] = ft_notlast
    learn_this_layer['inception_5a_3x3'] = ft_notlast
    learn_this_layer['inception_5a_5x5_reduce'] = ft_notlast
    learn_this_layer['inception_5a_5x5'] = ft_notlast
    learn_this_layer['inception_5a_pool_proj'] = ft_notlast

    learn_this_layer['inception_5b_1x1'] = ft_notlast
    learn_this_layer['inception_5b_3x3_reduce'] = ft_notlast
    learn_this_layer['inception_5b_3x3'] = ft_notlast
    learn_this_layer['inception_5b_5x5_reduce'] = ft_notlast
    learn_this_layer['inception_5b_5x5'] = ft_notlast
    learn_this_layer['inception_5b_pool_proj'] = ft_notlast

    learn_this_layer['loss1_conv'] = ft_notlast
    learn_this_layer['loss1_fc'] = ft_notlast
    learn_this_layer['loss1_classifier'] = 1

    learn_this_layer['loss2_conv'] = ft_notlast
    learn_this_layer['loss2_fc'] = ft_notlast
    learn_this_layer['loss2_classifier'] = 1

    learn_this_layer['loss3_classifier'] = 1

    return learn_this_layer


def learn_layers_vgg(control=dict()):
    if 'f_freeze' in control.keys():
        if control['f_freeze'] == 'notlast':
            ft_notlast = 0
        else:
            ft_notlast = 1
    else:
        ft_notlast = 1

    learn_this_layer = {}
    learn_this_layer['conv1_1'] = ft_notlast
    learn_this_layer['conv1_2'] = ft_notlast
    learn_this_layer['conv2_1'] = ft_notlast
    learn_this_layer['conv2_2'] = ft_notlast

    learn_this_layer['conv3_1'] = ft_notlast
    learn_this_layer['conv3_2'] = ft_notlast
    learn_this_layer['conv3_3'] = ft_notlast

    learn_this_layer['conv4_1'] = ft_notlast
    learn_this_layer['conv4_2'] = ft_notlast
    learn_this_layer['conv4_3'] = ft_notlast

    learn_this_layer['conv5_1'] = ft_notlast
    learn_this_layer['conv5_2'] = ft_notlast
    learn_this_layer['conv5_3'] = ft_notlast

    learn_this_layer['fc6'] = ft_notlast
    learn_this_layer['fc7'] = ft_notlast
    learn_this_layer['fc8'] = 1

    return learn_this_layer


def learn_layers_alexnet(control=dict()):
    if 'f_freeze' in control.keys():
        if control['f_freeze'] == 'notlast':
            ft_notlast = 0
        else:
            ft_notlast = 1
    else:
        ft_notlast = 1

    learn_this_layer = {}
    learn_this_layer['conv1'] = ft_notlast
    learn_this_layer['conv2'] = ft_notlast
    learn_this_layer['conv3'] = ft_notlast
    learn_this_layer['conv4'] = ft_notlast
    learn_this_layer['conv5'] = ft_notlast
    learn_this_layer['fc6'] = ft_notlast
    learn_this_layer['fc7'] = ft_notlast
    learn_this_layer['fc8'] = 1

    return learn_this_layer


def resnet_layers(n, is_train, control, conf, nlayers):
    if 'f_freeze' in control.keys():
        if control['f_freeze'] == 'notlast':
            ft_notlast = 0
        else:
            ft_notlast = 1
    else:
        ft_notlast = 1

    learned_param_nobias = [
        dict(lr_mult=0, decay_mult=0),
        dict(lr_mult=1, decay_mult=1),
    ]
    learned_param_bias = [
        [dict(lr_mult=0, decay_mult=0),
         dict(lr_mult=0, decay_mult=0)],
        [dict(lr_mult=1, decay_mult=1),
         dict(lr_mult=2, decay_mult=0)],
    ]

    weight_filler = dict(type='msra')
    bias_filler = dict(type='constant', value=0)

    if is_train:
        use_global_stats = False
    else:
        use_global_stats = True

    ##


    def ConvBnScaleRelu(n, nameset, no, ks, p, s, bias, bottom):

        if bias:
            params_conv = dict(name=nameset[0],
                               convolution_param=dict(num_output=no, kernel_size=ks, pad=p, stride=s, bias_term=True,
                                                      weight_filler=weight_filler,
                                                      bias_filler=bias_filler),
                               param=learned_param_bias[ft_notlast],

                               )
        else:
            params_conv = dict(name=nameset[0],
                               convolution_param=dict(num_output=no, kernel_size=ks, pad=p, stride=s, bias_term=False,
                                                      weight_filler=weight_filler),
                               param=learned_param_nobias[ft_notlast],
                               )
        params_bn = dict(name=nameset[1], batch_norm_param=dict(use_global_stats=use_global_stats), in_place=True
                         )
        params_scale = dict(name=nameset[2], scale_param=dict(bias_term=True), in_place=True
                            )
        params_relu = dict(name=nameset[3], in_place=True)

        conv = BaseLegoFunction('Convolution', params_conv).attach(n, bottom)
        bn = BaseLegoFunction('BatchNorm', params_bn).attach(n, [conv])
        scale = BaseLegoFunction('Scale', params_scale).attach(n, [bn])
        relu = BaseLegoFunction('ReLU', params_relu).attach(n, [scale])
        return relu

    def ConvBnScale(n, nameset, no, ks, p, s, bias, bottom):

        if bias:
            params_conv = dict(name=nameset[0],
                               convolution_param=dict(num_output=no, kernel_size=ks, pad=p, stride=s, bias_term=True,
                                                      weight_filler=weight_filler,
                                                      bias_filler=bias_filler),
                               param=learned_param_bias[ft_notlast],
                               )
        else:
            params_conv = dict(name=nameset[0],
                               convolution_param=dict(num_output=no, kernel_size=ks, pad=p, stride=s, bias_term=False,
                                                      weight_filler=weight_filler),
                               param=learned_param_nobias[ft_notlast],
                               )
        params_bn = dict(name=nameset[1], batch_norm_param=dict(use_global_stats=use_global_stats), in_place=True
                         )
        params_scale = dict(name=nameset[2], scale_param=dict(bias_term=True), in_place=True
                            )
        conv = BaseLegoFunction('Convolution', params_conv).attach(n, bottom)
        bn = BaseLegoFunction('BatchNorm', params_bn).attach(n, [conv])
        scale = BaseLegoFunction('Scale', params_scale).attach(n, [bn])
        return scale

    def EltRelu(n, nameset, bottom):
        params_elt = dict(name=nameset[0])
        params_relu = dict(name=nameset[1], in_place=True)
        elt = BaseLegoFunction('Eltwise', params_elt).attach(n, bottom)
        relu = BaseLegoFunction('ReLU', params_relu).attach(n, [elt])
        return relu

    def ThreeBlock(n, stagename, no, s, bottom):
        nameset = ['res' + stagename + '_branch2a', 'bn' + stagename + '_branch2a', 'scale' + stagename + '_branch2a',
                   'res' + stagename + '_branch2a_relu']
        branch2a = ConvBnScaleRelu(n, nameset, no=no, ks=1, p=0, s=s, bias=False, bottom=bottom)

        nameset = ['res' + stagename + '_branch2b', 'bn' + stagename + '_branch2b', 'scale' + stagename + '_branch2b',
                   'res' + stagename + '_branch2b_relu']
        branch2b = ConvBnScaleRelu(n, nameset, no=no, ks=3, p=1, s=1, bias=False, bottom=[branch2a])

        nameset = ['res' + stagename + '_branch2c', 'bn' + stagename + '_branch2c', 'scale' + stagename + '_branch2c']
        out = ConvBnScale(n, nameset, no=no * 4, ks=1, p=0, s=1, bias=False, bottom=[branch2b])
        return out

    def ProjRes(n, stagename, no, s, bottom):
        nameset = ['res' + stagename + '_branch1', 'bn' + stagename + '_branch1', 'scale' + stagename + '_branch1']
        branch1 = ConvBnScale(n, nameset, no=no * 4, ks=1, p=0, s=s, bias=False, bottom=bottom)
        branch2 = ThreeBlock(n, stagename, no=no, s=s, bottom=bottom)

        nameset = ['res' + stagename, 'res' + stagename + '_relu']
        out = EltRelu(n, nameset, bottom=[branch1, branch2])
        return out

    def IdRes(n, stagename, no, s, bottom):
        assert (s == 1)
        branch1 = bottom[0]
        branch2 = ThreeBlock(n, stagename, no=no, s=s, bottom=bottom)

        nameset = ['res' + stagename, 'res' + stagename + '_relu']
        out = EltRelu(n, nameset, bottom=[branch1, branch2])
        return out

    if nlayers == 50:
        stage1bias = True
        stagelist = [
            ['2a', 1, 64, 1],
            ['2b', 0, 64, 1],
            ['2c', 0, 64, 1],
            ['3a', 1, 128, 2],
            ['3b', 0, 128, 1],
            ['3c', 0, 128, 1],
            ['3d', 0, 128, 1],
            ['4a', 1, 256, 2],
            ['4b', 0, 256, 1],
            ['4c', 0, 256, 1],
            ['4d', 0, 256, 1],
            ['4e', 0, 256, 1],
            ['4f', 0, 256, 1],
            ['5a', 1, 512, 2],
            ['5b', 0, 512, 1],
            ['5c', 0, 512, 1],
        ]
    elif nlayers == 101:
        stage1bias = False
        stagelist = [
            ['2a', 1, 64, 1],
            ['2b', 0, 64, 1],
            ['2c', 0, 64, 1],
            ['3a', 1, 128, 2],
            ['3b1', 0, 128, 1],
            ['3b2', 0, 128, 1],
            ['3b3', 0, 128, 1],
            ['4a', 1, 256, 2],
            ['4b1', 0, 256, 1],
            ['4b2', 0, 256, 1],
            ['4b3', 0, 256, 1],
            ['4b4', 0, 256, 1],
            ['4b5', 0, 256, 1],
            ['4b6', 0, 256, 1],
            ['4b7', 0, 256, 1],
            ['4b8', 0, 256, 1],
            ['4b9', 0, 256, 1],
            ['4b10', 0, 256, 1],
            ['4b11', 0, 256, 1],
            ['4b12', 0, 256, 1],
            ['4b13', 0, 256, 1],
            ['4b14', 0, 256, 1],
            ['4b15', 0, 256, 1],
            ['4b16', 0, 256, 1],
            ['4b17', 0, 256, 1],
            ['4b18', 0, 256, 1],
            ['4b19', 0, 256, 1],
            ['4b20', 0, 256, 1],
            ['4b21', 0, 256, 1],
            ['4b22', 0, 256, 1],
            ['5a', 1, 512, 2],
            ['5b', 0, 512, 1],
            ['5c', 0, 512, 1],
        ]
    elif nlayers == 152:
        stage1bias = False
        stagelist = [
            ['2a', 1, 64, 1],
            ['2b', 0, 64, 1],
            ['2c', 0, 64, 1],
            ['3a', 1, 128, 2],
            ['3b1', 0, 128, 1],
            ['3b2', 0, 128, 1],
            ['3b3', 0, 128, 1],
            ['3b4', 0, 128, 1],
            ['3b5', 0, 128, 1],
            ['3b6', 0, 128, 1],
            ['3b7', 0, 128, 1],
            ['4a', 1, 256, 2],
            ['4b1', 0, 256, 1],
            ['4b2', 0, 256, 1],
            ['4b3', 0, 256, 1],
            ['4b4', 0, 256, 1],
            ['4b5', 0, 256, 1],
            ['4b6', 0, 256, 1],
            ['4b7', 0, 256, 1],
            ['4b8', 0, 256, 1],
            ['4b9', 0, 256, 1],
            ['4b10', 0, 256, 1],
            ['4b11', 0, 256, 1],
            ['4b12', 0, 256, 1],
            ['4b13', 0, 256, 1],
            ['4b14', 0, 256, 1],
            ['4b15', 0, 256, 1],
            ['4b16', 0, 256, 1],
            ['4b17', 0, 256, 1],
            ['4b18', 0, 256, 1],
            ['4b19', 0, 256, 1],
            ['4b20', 0, 256, 1],
            ['4b21', 0, 256, 1],
            ['4b22', 0, 256, 1],
            ['4b23', 0, 256, 1],
            ['4b24', 0, 256, 1],
            ['4b25', 0, 256, 1],
            ['4b26', 0, 256, 1],
            ['4b27', 0, 256, 1],
            ['4b28', 0, 256, 1],
            ['4b29', 0, 256, 1],
            ['4b30', 0, 256, 1],
            ['4b31', 0, 256, 1],
            ['4b32', 0, 256, 1],
            ['4b33', 0, 256, 1],
            ['4b34', 0, 256, 1],
            ['4b35', 0, 256, 1],
            ['5a', 1, 512, 2],
            ['5b', 0, 512, 1],
            ['5c', 0, 512, 1],
        ]
    else:
        raise

    # Stage 1

    nameset = ['conv1', 'bn_conv1', 'scale_conv1', 'conv1_relu']
    relu = ConvBnScaleRelu(n, nameset, no=64, ks=7, p=3, s=2, bias=stage1bias, bottom=[n.data])
    params_pool = dict(pooling_param=dict(kernel_size=3, stride=2, pool=P.Pooling.MAX), name='pool1')
    stage1 = BaseLegoFunction('Pooling', params_pool).attach(n, [relu])

    prev = stage1

    for st in stagelist:
        st_name, projection, no, s = st
        if projection:
            next = ProjRes(n, st_name, no=no, s=s, bottom=[prev])
        else:
            next = IdRes(n, st_name, no=no, s=s, bottom=[prev])

        prev = next

    finalstage = next

    # final
    params_pool = dict(pooling_param=dict(kernel_size=7, stride=1, pool=P.Pooling.AVE), name='pool5')
    pool5 = BaseLegoFunction('Pooling', params_pool).attach(n, [finalstage])

    if 'secondtime' in conf.keys():
        params_fc = dict(inner_product_param=dict(num_output=conf['f_nout']), name=conf['f_outname'])
    else:
        params_fc = dict(inner_product_param=dict(num_output=conf['nout']), name=conf['outname'])
    score = BaseLegoFunction('InnerProduct', params_fc).attach(n, [pool5])

    return n, score


def googlenet_layers(n, conf, learn_this_layer=learn_layers_googlenet()):
    param = (get_frozen_param(), get_learned_param())

    # 1
    n.conv1_7x7_s2, n.conv1_relu_7x7 = conv_relu(n.data, ks=7, nout=64, pad=3, stride=2, in_place=True,
                                                 weight_filler=dict(type='xavier'),
                                                 bias_filler=dict(type='constant', value=0.2),
                                                 param=param[learn_this_layer['conv1_7x7_s2']])
    n.pool1_3x3_s2 = max_pool(n.conv1_relu_7x7, ks=3, stride=2)
    n.pool1_norm1 = lrn(n.pool1_3x3_s2, local_size=5, alpha=0.0001, beta=0.75)

    # 2
    n.conv2_3x3_reduce, n.conv2_relu_3x3_reduce = conv_relu(n.pool1_norm1, ks=1, nout=64, in_place=True,
                                                            weight_filler=dict(type='xavier'),
                                                            bias_filler=dict(type='constant', value=0.2),
                                                            param=param[learn_this_layer['conv2_3x3_reduce']])
    n.conv2_3x3, n.conv2_relu3x3 = conv_relu(n.conv2_relu_3x3_reduce, nout=192, pad=1, ks=3, in_place=True,
                                             weight_filler=dict(type='xavier'),
                                             bias_filler=dict(type='constant', value=0.2),
                                             param=param[learn_this_layer['conv2_3x3']])
    n.conv2_norm2 = lrn(n.conv2_relu3x3, local_size=5, alpha=0.0001, beta=0.75)
    n.pool2_3x3_s2 = max_pool(n.conv2_norm2, ks=3, stride=2)

    # 3a
    n.inception_3a_1x1, n.inception_3a_relu_1x1 = conv_relu(n.pool2_3x3_s2, ks=1, nout=64, in_place=True,
                                                            weight_filler=dict(type='xavier'),
                                                            bias_filler=dict(type='constant', value=0.2),
                                                            param=param[learn_this_layer['inception_3a_1x1']])
    n.inception_3a_3x3_reduce, n.inception_3a_relu_3x3_reduce = conv_relu(n.pool2_3x3_s2, ks=1, nout=96, in_place=True,
                                                                          weight_filler=dict(type='xavier'),
                                                                          bias_filler=dict(type='constant', value=0.2),
                                                                          param=param[learn_this_layer[
                                                                              'inception_3a_3x3_reduce']])
    n.inception_3a_3x3, n.inception_3a_relu_3x3 = conv_relu(n.inception_3a_relu_3x3_reduce,
                                                            ks=3, nout=128, pad=1, in_place=True,
                                                            weight_filler=dict(type='xavier'),
                                                            bias_filler=dict(type='constant', value=0.2),
                                                            param=param[learn_this_layer['inception_3a_3x3']])
    n.inception_3a_5x5_reduce, n.inception_3a_relu_5x5_reduce = conv_relu(n.pool2_3x3_s2, ks=1, nout=16, in_place=True,
                                                                          param=param[learn_this_layer[
                                                                              'inception_3a_5x5_reduce']])
    n.inception_3a_5x5, n.inception_3a_relu_5x5 = conv_relu(n.inception_3a_relu_5x5_reduce,
                                                            ks=5, nout=32, pad=2, in_place=True,
                                                            weight_filler=dict(type='xavier'),
                                                            bias_filler=dict(type='constant', value=0.2),
                                                            param=param[learn_this_layer['inception_3a_5x5']])
    n.inception_3a_pool = max_pool(n.pool2_3x3_s2, ks=3, stride=1, pad=1)

    n.inception_3a_pool_proj, n.inception_3a_relu_pool_proj = conv_relu(n.inception_3a_pool, ks=1, nout=32,
                                                                        in_place=True,
                                                                        weight_filler=dict(type='xavier'),
                                                                        bias_filler=dict(type='constant', value=0.2),
                                                                        param=param[learn_this_layer[
                                                                            'inception_3a_pool_proj']])
    n.inception_3a_output = L.Concat(n.inception_3a_relu_1x1, n.inception_3a_relu_3x3, n.inception_3a_relu_5x5,
                                     n.inception_3a_relu_pool_proj)

    # 3b
    n.inception_3b_1x1, n.inception_3b_relu_1x1 = conv_relu(n.inception_3a_output, ks=1, nout=128, in_place=True,
                                                            weight_filler=dict(type='xavier'),
                                                            bias_filler=dict(type='constant', value=0.2),
                                                            param=param[learn_this_layer['inception_3b_1x1']])
    n.inception_3b_3x3_reduce, n.inception_3b_relu_3x3_reduce = conv_relu(n.inception_3a_output, ks=1, nout=128,
                                                                          in_place=True,
                                                                          weight_filler=dict(type='xavier'),
                                                                          bias_filler=dict(type='constant', value=0.2),
                                                                          param=param[learn_this_layer[
                                                                              'inception_3b_3x3_reduce']])
    n.inception_3b_3x3, n.inception_3b_relu_3x3 = conv_relu(n.inception_3b_relu_3x3_reduce,
                                                            ks=3, nout=192, pad=1, in_place=True,
                                                            weight_filler=dict(type='xavier'),
                                                            bias_filler=dict(type='constant', value=0.2),
                                                            param=param[learn_this_layer['inception_3b_3x3']])
    n.inception_3b_5x5_reduce, n.inception_3b_relu_5x5_reduce = conv_relu(n.inception_3a_output, ks=1, nout=32,
                                                                          in_place=True,
                                                                          weight_filler=dict(type='xavier'),
                                                                          bias_filler=dict(type='constant', value=0.2),
                                                                          param=param[learn_this_layer[
                                                                              'inception_3b_5x5_reduce']])
    n.inception_3b_5x5, n.inception_3b_relu_5x5 = conv_relu(n.inception_3b_relu_5x5_reduce, in_place=True,
                                                            weight_filler=dict(type='xavier'),
                                                            bias_filler=dict(type='constant', value=0.2),
                                                            ks=5, nout=96, pad=2,
                                                            param=param[learn_this_layer['inception_3b_5x5']])
    n.inception_3b_pool = max_pool(n.inception_3a_output, ks=3, stride=1, pad=1)

    n.inception_3b_pool_proj, n.inception_3b_relu_pool_proj = conv_relu(n.inception_3b_pool, ks=1, nout=64,
                                                                        in_place=True,
                                                                        weight_filler=dict(type='xavier'),
                                                                        bias_filler=dict(type='constant', value=0.2),
                                                                        param=param[learn_this_layer[
                                                                            'inception_3b_pool_proj']])
    n.inception_3b_output = L.Concat(n.inception_3b_relu_1x1, n.inception_3b_relu_3x3, n.inception_3b_relu_5x5,
                                     n.inception_3b_relu_pool_proj)

    n.pool3_3x3_s2 = max_pool(n.inception_3b_output, ks=3, stride=2)

    # 4a
    n.inception_4a_1x1, n.inception_4a_relu_1x1 = conv_relu(n.pool3_3x3_s2, ks=1, nout=192, in_place=True,
                                                            weight_filler=dict(type='xavier'),
                                                            bias_filler=dict(type='constant', value=0.2),
                                                            param=param[learn_this_layer['inception_4a_1x1']])
    n.inception_4a_3x3_reduce, n.inception_4a_relu_3x3_reduce = conv_relu(n.pool3_3x3_s2, ks=1, nout=96, in_place=True,
                                                                          weight_filler=dict(type='xavier'),
                                                                          bias_filler=dict(type='constant', value=0.2),
                                                                          param=param[learn_this_layer[
                                                                              'inception_4a_3x3_reduce']])
    n.inception_4a_3x3, n.inception_4a_relu_3x3 = conv_relu(n.inception_4a_relu_3x3_reduce,
                                                            ks=3, nout=208, pad=1, in_place=True,
                                                            weight_filler=dict(type='xavier'),
                                                            bias_filler=dict(type='constant', value=0.2),
                                                            param=param[learn_this_layer['inception_4a_3x3']])
    n.inception_4a_5x5_reduce, n.inception_4a_relu_5x5_reduce = conv_relu(n.pool3_3x3_s2, ks=1, nout=16, in_place=True,
                                                                          weight_filler=dict(type='xavier'),
                                                                          bias_filler=dict(type='constant', value=0.2),
                                                                          param=param[learn_this_layer[
                                                                              'inception_4a_5x5_reduce']])
    n.inception_4a_5x5, n.inception_4a_relu_5x5 = conv_relu(n.inception_4a_relu_5x5_reduce,
                                                            weight_filler=dict(type='xavier'),
                                                            bias_filler=dict(type='constant', value=0.2),
                                                            ks=5, nout=48, pad=2, in_place=True,
                                                            param=param[learn_this_layer['inception_4a_5x5']])
    n.inception_4a_pool = max_pool(n.pool3_3x3_s2, ks=3, stride=1, pad=1)

    n.inception_4a_pool_proj, n.inception_4a_relu_pool_proj = conv_relu(n.inception_4a_pool, ks=1, nout=64,
                                                                        in_place=True,
                                                                        weight_filler=dict(type='xavier'),
                                                                        bias_filler=dict(type='constant', value=0.2),
                                                                        param=param[learn_this_layer[
                                                                            'inception_4a_pool_proj']])
    n.inception_4a_output = L.Concat(n.inception_4a_relu_1x1, n.inception_4a_relu_3x3, n.inception_4a_relu_5x5,
                                     n.inception_4a_relu_pool_proj)

    # l1

    n.loss1_ave_pool = L.Pooling(n.inception_4a_output, pooling_param=dict(pool=P.Pooling.AVE, kernel_size=5, stride=3))

    n.loss1_conv, n.loss1_relu_conv = conv_relu(n.loss1_ave_pool, nout=128, ks=1, in_place=True,
                                                weight_filler=dict(type='xavier'),
                                                bias_filler=dict(type='constant', value=0.2),
                                                param=param[learn_this_layer['loss1_conv']])

    n.loss1_fc, n.loss1_relu_fc = fc_relu(n.loss1_relu_conv, nout=1024, in_place=True,
                                          weight_filler=dict(type='xavier'),
                                          bias_filler=dict(type='constant', value=0.2),
                                          param=param[learn_this_layer['loss1_fc']])
    n.loss1_drop_fc = L.Dropout(n.loss1_relu_fc, dropout_param=dict(dropout_ratio=0.7), in_place=True)

    if 'secondtime' in conf.keys():
        params_fc = dict(inner_product_param=dict(num_output=conf['f_nout'],
                                                  weight_filler=dict(type='xavier'),
                                                  bias_filler=dict(type='constant', value=0)),
                         param=param[learn_this_layer['loss1_classifier']],
                         name=conf['f_outname1'])
    else:
        params_fc = dict(inner_product_param=dict(num_output=conf['nout'],
                                                  weight_filler=dict(type='xavier'),
                                                  bias_filler=dict(type='constant', value=0)),
                         param=param[learn_this_layer['loss1_classifier']],
                         name=conf['outname1'])
    score1 = BaseLegoFunction('InnerProduct', params_fc).attach(n, [n.loss1_drop_fc])

    # 4b
    n.inception_4b_1x1, n.inception_4b_relu_1x1 = conv_relu(n.inception_4a_output, ks=1, nout=160, in_place=True,
                                                            weight_filler=dict(type='xavier'),
                                                            bias_filler=dict(type='constant', value=0.2),
                                                            param=param[learn_this_layer['inception_4b_1x1']])
    n.inception_4b_3x3_reduce, n.inception_4b_relu_3x3_reduce = conv_relu(n.inception_4a_output, ks=1, nout=112,
                                                                          in_place=True,
                                                                          weight_filler=dict(type='xavier'),
                                                                          bias_filler=dict(type='constant', value=0.2),
                                                                          param=param[learn_this_layer[
                                                                              'inception_4b_3x3_reduce']])
    n.inception_4b_3x3, n.inception_4b_relu_3x3 = conv_relu(n.inception_4b_relu_3x3_reduce,
                                                            ks=3, nout=224, pad=1, in_place=True,
                                                            weight_filler=dict(type='xavier'),
                                                            bias_filler=dict(type='constant', value=0.2),
                                                            param=param[learn_this_layer['inception_4b_3x3']])
    n.inception_4b_5x5_reduce, n.inception_4b_relu_5x5_reduce = conv_relu(n.inception_4a_output, ks=1, nout=24,
                                                                          in_place=True,
                                                                          weight_filler=dict(type='xavier'),
                                                                          bias_filler=dict(type='constant', value=0.2),
                                                                          param=param[learn_this_layer[
                                                                              'inception_4b_5x5_reduce']])
    n.inception_4b_5x5, n.inception_4b_relu_5x5 = conv_relu(n.inception_4b_relu_5x5_reduce,
                                                            ks=5, nout=64, pad=2, in_place=True,
                                                            weight_filler=dict(type='xavier'),
                                                            bias_filler=dict(type='constant', value=0.2),
                                                            param=param[learn_this_layer['inception_4b_5x5']])
    n.inception_4b_pool = max_pool(n.inception_4a_output, ks=3, stride=1, pad=1)

    n.inception_4b_pool_proj, n.inception_4b_relu_pool_proj = conv_relu(n.inception_4b_pool, ks=1, nout=64,
                                                                        in_place=True,
                                                                        weight_filler=dict(type='xavier'),
                                                                        bias_filler=dict(type='constant', value=0.2),
                                                                        param=param[learn_this_layer[
                                                                            'inception_4b_pool_proj']])
    n.inception_4b_output = L.Concat(n.inception_4b_relu_1x1, n.inception_4b_relu_3x3, n.inception_4b_relu_5x5,
                                     n.inception_4b_relu_pool_proj)

    # 4c
    n.inception_4c_1x1, n.inception_4c_relu_1x1 = conv_relu(n.inception_4b_output, ks=1, nout=128, in_place=True,
                                                            weight_filler=dict(type='xavier'),
                                                            bias_filler=dict(type='constant', value=0.2),
                                                            param=param[learn_this_layer['inception_4c_1x1']])
    n.inception_4c_3x3_reduce, n.inception_4c_relu_3x3_reduce = conv_relu(n.inception_4b_output, ks=1, nout=128,
                                                                          in_place=True,
                                                                          weight_filler=dict(type='xavier'),
                                                                          bias_filler=dict(type='constant', value=0.2),
                                                                          param=param[learn_this_layer[
                                                                              'inception_4c_3x3_reduce']])
    n.inception_4c_3x3, n.inception_4c_relu_3x3 = conv_relu(n.inception_4c_relu_3x3_reduce,
                                                            ks=3, nout=256, pad=1, in_place=True,
                                                            weight_filler=dict(type='xavier'),
                                                            bias_filler=dict(type='constant', value=0.2),
                                                            param=param[learn_this_layer['inception_4c_3x3']])
    n.inception_4c_5x5_reduce, n.inception_4c_relu_5x5_reduce = conv_relu(n.inception_4b_output, ks=1, nout=24,
                                                                          in_place=True,
                                                                          weight_filler=dict(type='xavier'),
                                                                          bias_filler=dict(type='constant', value=0.2),
                                                                          param=param[learn_this_layer[
                                                                              'inception_4c_5x5_reduce']])
    n.inception_4c_5x5, n.inception_4c_relu_5x5 = conv_relu(n.inception_4c_relu_5x5_reduce,
                                                            ks=5, nout=64, pad=2, in_place=True,
                                                            weight_filler=dict(type='xavier'),
                                                            bias_filler=dict(type='constant', value=0.2),
                                                            param=param[learn_this_layer['inception_4c_5x5']])
    n.inception_4c_pool = max_pool(n.inception_4b_output, ks=3, stride=1, pad=1)

    n.inception_4c_pool_proj, n.inception_4c_relu_pool_proj = conv_relu(n.inception_4c_pool, ks=1, nout=64,
                                                                        in_place=True,
                                                                        weight_filler=dict(type='xavier'),
                                                                        bias_filler=dict(type='constant', value=0.2),
                                                                        param=param[learn_this_layer[
                                                                            'inception_4c_pool_proj']])
    n.inception_4c_output = L.Concat(n.inception_4c_relu_1x1, n.inception_4c_relu_3x3, n.inception_4c_relu_5x5,
                                     n.inception_4c_relu_pool_proj)

    # 4d
    n.inception_4d_1x1, n.inception_4d_relu_1x1 = conv_relu(n.inception_4c_output, ks=1, nout=112, in_place=True,
                                                            weight_filler=dict(type='xavier'),
                                                            bias_filler=dict(type='constant', value=0.2),
                                                            param=param[learn_this_layer['inception_4d_1x1']])
    n.inception_4d_3x3_reduce, n.inception_4d_relu_3x3_reduce = conv_relu(n.inception_4c_output, ks=1, nout=144,
                                                                          in_place=True,
                                                                          weight_filler=dict(type='xavier'),
                                                                          bias_filler=dict(type='constant', value=0.2),
                                                                          param=param[learn_this_layer[
                                                                              'inception_4d_3x3_reduce']])
    n.inception_4d_3x3, n.inception_4d_relu_3x3 = conv_relu(n.inception_4d_relu_3x3_reduce,
                                                            ks=3, nout=288, pad=1, in_place=True,
                                                            weight_filler=dict(type='xavier'),
                                                            bias_filler=dict(type='constant', value=0.2),
                                                            param=param[learn_this_layer['inception_4d_3x3']])
    n.inception_4d_5x5_reduce, n.inception_4d_relu_5x5_reduce = conv_relu(n.inception_4c_output, ks=1, nout=32,
                                                                          in_place=True,
                                                                          weight_filler=dict(type='xavier'),
                                                                          bias_filler=dict(type='constant', value=0.2),
                                                                          param=param[learn_this_layer[
                                                                              'inception_4d_5x5_reduce']])
    n.inception_4d_5x5, n.inception_4d_relu_5x5 = conv_relu(n.inception_4d_relu_5x5_reduce,
                                                            ks=5, nout=64, pad=2, in_place=True,
                                                            weight_filler=dict(type='xavier'),
                                                            bias_filler=dict(type='constant', value=0.2),
                                                            param=param[learn_this_layer['inception_4d_5x5']])
    n.inception_4d_pool = max_pool(n.inception_4c_output, ks=3, stride=1, pad=1)

    n.inception_4d_pool_proj, n.inception_4d_relu_pool_proj = conv_relu(n.inception_4d_pool, ks=1, nout=64,
                                                                        in_place=True,
                                                                        weight_filler=dict(type='xavier'),
                                                                        bias_filler=dict(type='constant', value=0.2),
                                                                        param=param[learn_this_layer[
                                                                            'inception_4d_pool_proj']])
    n.inception_4d_output = L.Concat(n.inception_4d_relu_1x1, n.inception_4d_relu_3x3, n.inception_4d_relu_5x5,
                                     n.inception_4d_relu_pool_proj)

    # l2

    n.loss2_ave_pool = L.Pooling(n.inception_4d_output, pooling_param=dict(pool=P.Pooling.AVE, kernel_size=5, stride=3))

    n.loss2_conv, n.loss2_relu_conv = conv_relu(n.loss2_ave_pool, nout=128, ks=1, in_place=True,
                                                weight_filler=dict(type='xavier'),
                                                bias_filler=dict(type='constant', value=0.2),
                                                param=param[learn_this_layer['loss2_conv']])

    n.loss2_fc, n.loss2_relu_fc = fc_relu(n.loss2_relu_conv, nout=1024, in_place=True,
                                          weight_filler=dict(type='xavier'),
                                          bias_filler=dict(type='constant', value=0.2),
                                          param=param[learn_this_layer['loss2_fc']])
    n.loss2_drop_fc = L.Dropout(n.loss2_relu_fc, dropout_param=dict(dropout_ratio=0.7), in_place=True)

    if 'secondtime' in conf.keys():
        params_fc = dict(inner_product_param=dict(num_output=conf['f_nout'],
                                                  weight_filler=dict(type='xavier'),
                                                  bias_filler=dict(type='constant', value=0)),
                         param=param[learn_this_layer['loss2_classifier']],
                         name=conf['f_outname2'])
    else:
        params_fc = dict(inner_product_param=dict(num_output=conf['nout'],
                                                  weight_filler=dict(type='xavier'),
                                                  bias_filler=dict(type='constant', value=0)),
                         param=param[learn_this_layer['loss2_classifier']],
                         name=conf['outname2'])
    score2 = BaseLegoFunction('InnerProduct', params_fc).attach(n, [n.loss2_drop_fc])

    # 4e
    n.inception_4e_1x1, n.inception_4e_relu_1x1 = conv_relu(n.inception_4d_output, ks=1, nout=256, in_place=True,
                                                            weight_filler=dict(type='xavier'),
                                                            bias_filler=dict(type='constant', value=0.2),
                                                            param=param[learn_this_layer['inception_4e_1x1']])
    n.inception_4e_3x3_reduce, n.inception_4e_relu_3x3_reduce = conv_relu(n.inception_4d_output, ks=1, nout=160,
                                                                          in_place=True,
                                                                          weight_filler=dict(type='xavier'),
                                                                          bias_filler=dict(type='constant', value=0.2),
                                                                          param=param[learn_this_layer[
                                                                              'inception_4e_3x3_reduce']])
    n.inception_4e_3x3, n.inception_4e_relu_3x3 = conv_relu(n.inception_4e_relu_3x3_reduce,
                                                            ks=3, nout=320, pad=1, in_place=True,
                                                            weight_filler=dict(type='xavier'),
                                                            bias_filler=dict(type='constant', value=0.2),
                                                            param=param[learn_this_layer['inception_4e_3x3']])
    n.inception_4e_5x5_reduce, n.inception_4e_relu_5x5_reduce = conv_relu(n.inception_4d_output, ks=1, nout=32,
                                                                          in_place=True,
                                                                          weight_filler=dict(type='xavier'),
                                                                          bias_filler=dict(type='constant', value=0.2),
                                                                          param=param[learn_this_layer[
                                                                              'inception_4e_5x5_reduce']])
    n.inception_4e_5x5, n.inception_4e_relu_5x5 = conv_relu(n.inception_4e_relu_5x5_reduce,
                                                            ks=5, nout=128, pad=2, in_place=True,
                                                            weight_filler=dict(type='xavier'),
                                                            bias_filler=dict(type='constant', value=0.2),
                                                            param=param[learn_this_layer['inception_4e_5x5']])
    n.inception_4e_pool = max_pool(n.inception_4d_output, ks=3, stride=1, pad=1)

    n.inception_4e_pool_proj, n.inception_4e_relu_pool_proj = conv_relu(n.inception_4e_pool, ks=1, nout=128,
                                                                        in_place=True,
                                                                        weight_filler=dict(type='xavier'),
                                                                        bias_filler=dict(type='constant', value=0.2),
                                                                        param=param[learn_this_layer[
                                                                            'inception_4e_pool_proj']])
    n.inception_4d_output = L.Concat(n.inception_4e_relu_1x1, n.inception_4e_relu_3x3, n.inception_4e_relu_5x5,
                                     n.inception_4e_relu_pool_proj)

    n.pool4_3x3_s2 = max_pool(n.inception_4d_output, ks=3, stride=2)

    # 5a
    n.inception_5a_1x1, n.inception_5a_relu_1x1 = conv_relu(n.pool4_3x3_s2, ks=1, nout=256, in_place=True,
                                                            weight_filler=dict(type='xavier'),
                                                            bias_filler=dict(type='constant', value=0.2),
                                                            param=param[learn_this_layer['inception_5a_1x1']])
    n.inception_5a_3x3_reduce, n.inception_5a_relu_3x3_reduce = conv_relu(n.pool4_3x3_s2, ks=1, nout=160, in_place=True,
                                                                          weight_filler=dict(type='xavier'),
                                                                          bias_filler=dict(type='constant', value=0.2),
                                                                          param=param[learn_this_layer[
                                                                              'inception_5a_3x3_reduce']])
    n.inception_5a_3x3, n.inception_5a_relu_3x3 = conv_relu(n.inception_5a_relu_3x3_reduce,
                                                            ks=3, nout=320, pad=1, in_place=True,
                                                            weight_filler=dict(type='xavier'),
                                                            bias_filler=dict(type='constant', value=0.2),
                                                            param=param[learn_this_layer['inception_5a_3x3']])
    n.inception_5a_5x5_reduce, n.inception_5a_relu_5x5_reduce = conv_relu(n.pool4_3x3_s2, ks=1, nout=32, in_place=True,
                                                                          weight_filler=dict(type='xavier'),
                                                                          bias_filler=dict(type='constant', value=0.2),
                                                                          param=param[learn_this_layer[
                                                                              'inception_5a_5x5_reduce']])
    n.inception_5a_5x5, n.inception_5a_relu_5x5 = conv_relu(n.inception_5a_relu_5x5_reduce,
                                                            ks=5, nout=128, pad=2, in_place=True,
                                                            weight_filler=dict(type='xavier'),
                                                            bias_filler=dict(type='constant', value=0.2),
                                                            param=param[learn_this_layer['inception_5a_5x5']])
    n.inception_5a_pool = max_pool(n.pool4_3x3_s2, ks=3, stride=1, pad=1)

    n.inception_5a_pool_proj, n.inception_5a_relu_pool_proj = conv_relu(n.inception_5a_pool, ks=1, nout=128,
                                                                        in_place=True,
                                                                        weight_filler=dict(type='xavier'),
                                                                        bias_filler=dict(type='constant', value=0.2),
                                                                        param=param[learn_this_layer[
                                                                            'inception_5a_pool_proj']])
    n.inception_5a_output = L.Concat(n.inception_5a_relu_1x1, n.inception_5a_relu_3x3, n.inception_5a_relu_5x5,
                                     n.inception_5a_relu_pool_proj)

    # 5b
    n.inception_5b_1x1, n.inception_5b_relu_1x1 = conv_relu(n.inception_5a_output, ks=1, nout=384, in_place=True,
                                                            weight_filler=dict(type='xavier'),
                                                            bias_filler=dict(type='constant', value=0.2),
                                                            param=param[learn_this_layer['inception_5b_1x1']])
    n.inception_5b_3x3_reduce, n.inception_5b_relu_3x3_reduce = conv_relu(n.inception_5a_output, ks=1, nout=192,
                                                                          in_place=True,
                                                                          weight_filler=dict(type='xavier'),
                                                                          bias_filler=dict(type='constant', value=0.2),
                                                                          param=param[learn_this_layer[
                                                                              'inception_5b_3x3_reduce']])
    n.inception_5b_3x3, n.inception_5b_relu_3x3 = conv_relu(n.inception_5b_relu_3x3_reduce,
                                                            ks=3, nout=384, pad=1, in_place=True,
                                                            weight_filler=dict(type='xavier'),
                                                            bias_filler=dict(type='constant', value=0.2),
                                                            param=param[learn_this_layer['inception_5b_3x3']])
    n.inception_5b_5x5_reduce, n.inception_5b_relu_5x5_reduce = conv_relu(n.inception_5a_output, ks=1, nout=48,
                                                                          in_place=True,
                                                                          weight_filler=dict(type='xavier'),
                                                                          bias_filler=dict(type='constant', value=0.2),
                                                                          param=param[learn_this_layer[
                                                                              'inception_5b_5x5_reduce']])
    n.inception_5b_5x5, n.inception_5b_relu_5x5 = conv_relu(n.inception_5b_relu_5x5_reduce,
                                                            ks=5, nout=128, pad=2, in_place=True,
                                                            weight_filler=dict(type='xavier'),
                                                            bias_filler=dict(type='constant', value=0.2),
                                                            param=param[learn_this_layer['inception_5b_5x5']])
    n.inception_5b_pool = max_pool(n.inception_5a_output, ks=3, stride=1, pad=1)

    n.inception_5b_pool_proj, n.inception_5b_relu_pool_proj = conv_relu(n.inception_5b_pool, ks=1, nout=128,
                                                                        in_place=True,
                                                                        weight_filler=dict(type='xavier'),
                                                                        bias_filler=dict(type='constant', value=0.2),
                                                                        param=param[learn_this_layer[
                                                                            'inception_5b_pool_proj']])
    n.inception_5b_output = L.Concat(n.inception_5b_relu_1x1, n.inception_5b_relu_3x3, n.inception_5b_relu_5x5,
                                     n.inception_5b_relu_pool_proj)

    n.pool5_7x7_s1 = L.Pooling(n.inception_5b_output, pooling_param=dict(pool=P.Pooling.AVE, kernel_size=7, stride=1))

    n.pool5_drop_7x7_s1 = L.Dropout(n.pool5_7x7_s1, dropout_param=dict(dropout_ratio=0.4), in_place=True)

    # l3

    if 'secondtime' in conf.keys():
        params_fc = dict(inner_product_param=dict(num_output=conf['f_nout'],
                                                  weight_filler=dict(type='xavier'),
                                                  bias_filler=dict(type='constant', value=0)),
                         param=param[learn_this_layer['loss3_classifier']],
                         name=conf['f_outname3'])
    else:
        params_fc = dict(inner_product_param=dict(num_output=conf['nout'],
                                                  weight_filler=dict(type='xavier'),
                                                  bias_filler=dict(type='constant', value=0)),
                         param=param[learn_this_layer['loss3_classifier']],
                         name=conf['outname3'])
    score3 = BaseLegoFunction('InnerProduct', params_fc).attach(n, [n.pool5_drop_7x7_s1])

    return n, score1, score2, score3


def vgg_layers(n, conf, learn_this_layer=learn_layers_vgg()):
    param = (get_frozen_param(), get_learned_param())
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 3, 64, pad=1, param=param[learn_this_layer['conv1_1']])
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 3, 64, pad=1, param=param[learn_this_layer['conv1_2']])
    n.pool1 = max_pool(n.relu1_2, 2, stride=2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 3, 128, pad=1, param=param[learn_this_layer['conv2_1']])
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 3, 128, pad=1, param=param[learn_this_layer['conv2_2']])
    n.pool2 = max_pool(n.relu2_2, 2, stride=2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 3, 256, pad=1, param=param[learn_this_layer['conv3_1']])
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 3, 256, pad=1, param=param[learn_this_layer['conv3_2']])
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 3, 256, pad=1, param=param[learn_this_layer['conv3_3']])
    n.pool3 = max_pool(n.relu3_3, 2, stride=2)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 3, 512, pad=1, param=param[learn_this_layer['conv4_1']])
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 3, 512, pad=1, param=param[learn_this_layer['conv4_2']])
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 3, 512, pad=1, param=param[learn_this_layer['conv4_3']])
    n.pool4 = max_pool(n.relu4_3, 2, stride=2)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 3, 512, pad=1, param=param[learn_this_layer['conv5_1']])
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 3, 512, pad=1, param=param[learn_this_layer['conv5_2']])
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 3, 512, pad=1, param=param[learn_this_layer['conv5_3']])
    n.pool5 = max_pool(n.relu5_3, 2, stride=2)

    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=param[learn_this_layer['fc6']])
    n.drop6 = L.Dropout(n.relu6)
    n.fc7, n.relu7 = fc_relu(n.drop6, 4096, param=param[learn_this_layer['fc7']])
    n.drop7 = L.Dropout(n.relu7)

    if 'secondtime' in conf.keys():
        params_fc = dict(inner_product_param=dict(num_output=conf['f_nout']), param=param[learn_this_layer['fc8']],
                         name=conf['f_outname'])
    else:
        params_fc = dict(inner_product_param=dict(num_output=conf['nout']), param=param[learn_this_layer['fc8']],
                         name=conf['outname'])
        InnerProduct = dict(
            weight_filler=dict(type='msra'),
            bias_filler=dict(type='constant'),
            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
        )
    score = BaseLegoFunction('InnerProduct', params_fc).attach(n, [n.drop7])

    return n, score


def alexnet_layers(n, conf, learn_this_layer=learn_layers_alexnet()):
    param = (get_frozen_param(), get_learned_param())
    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4,
                                 weight_filler=dict(type='gaussian', std=0.01),
                                 bias_filler=dict(type='constant', value=0),
                                 param=param[learn_this_layer['conv1']])
    n.norm1 = lrn(n.relu1, local_size=5, alpha=0.0001, beta=0.75)
    n.pool1 = max_pool(n.norm1, 3, stride=2)

    n.conv2, n.relu2 = conv_relu(n.pool1, 5, 256, pad=2, group=2,
                                 weight_filler=dict(type='gaussian', std=0.01),
                                 bias_filler=dict(type='constant', value=0.1),
                                 param=param[learn_this_layer['conv2']])
    n.norm2 = lrn(n.relu2, local_size=5, alpha=0.0001, beta=0.75)
    n.pool2 = max_pool(n.norm2, 3, stride=2)

    n.conv3, n.relu3 = conv_relu(n.pool2, 3, 384, pad=1,
                                 weight_filler=dict(type='gaussian', std=0.01),
                                 bias_filler=dict(type='constant', value=0),
                                 param=param[learn_this_layer['conv3']])

    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2,
                                 weight_filler=dict(type='gaussian', std=0.01),
                                 bias_filler=dict(type='constant', value=0.1),
                                 param=param[learn_this_layer['conv4']])

    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2,
                                 weight_filler=dict(type='gaussian', std=0.01),
                                 bias_filler=dict(type='constant', value=0.1),
                                 param=param[learn_this_layer['conv5']])
    n.pool5 = max_pool(n.relu5, 3, 2)

    n.fc6, n.relu6 = fc_relu(n.pool5, 4096,
                             weight_filler=dict(type='gaussian', std=0.005),
                             bias_filler=dict(type='constant', value=0.1),
                             param=param[learn_this_layer['fc6']])
    n.drop6 = L.Dropout(n.relu6)
    n.fc7, n.relu7 = fc_relu(n.drop6, 4096,
                             weight_filler=dict(type='gaussian', std=0.005),
                             bias_filler=dict(type='constant', value=0.1),
                             param=param[learn_this_layer['fc7']])
    n.drop7 = L.Dropout(n.relu7)

    if 'secondtime' in conf.keys():
        params_fc = dict(
            name=conf['f_outname'],
            inner_product_param=dict(
                num_output=conf['f_nout'],
                weight_filler=dict(type='gaussian', std=0.01),
                bias_filler=dict(type='constant', value=0)
            ),
            param=[dict(lr_mult=10, decay_mult=1),
                   dict(lr_mult=20, decay_mult=0)],
        )
    else:
        params_fc = dict(
            name=conf['outname'],
            inner_product_param=dict(
                num_output=conf['nout'],
                weight_filler=dict(type='gaussian', std=0.01),
                bias_filler=dict(type='constant', value=0),
            ),
            param=[dict(lr_mult=10, decay_mult=1),
                   dict(lr_mult=20, decay_mult=0)],
        )
    score = BaseLegoFunction('InnerProduct', params_fc).attach(n, [n.drop7])

    return n, score


def nugu_test(conf, control, netname='net'):
    prefix = ''

    n = caffe.NetSpec()
    if 'resize_data' in conf.keys():
        n.data_ = L.DummyData(dummy_data_param=dict(num=1, channels=3, height=conf['inputsz'], width=conf['inputsz']))
        n.data = L.Python(n.data_,
                          python_param=dict(module='ssam_pylayers', layer='DataResizeLayer', param_str=str(dict(
                              num=1, channels=3, height=conf['inputsz'], width=conf['inputsz']))),
                          ntop=1, )
    else:
        n.data = L.DummyData(dummy_data_param=dict(num=1, channels=3, height=conf['inputsz'], width=conf['inputsz']))

    if 'ResNet' in control[netname]:
        n, score = resnet_layers(n, False, control, conf, int(control[netname][6:]))
        prob = BaseLegoFunction('Softmax', dict(name='prob')).attach(n, [score])
    elif control[netname] == 'GoogleNet':
        n, score1, score2, score3 = googlenet_layers(n, conf, learn_layers_googlenet())
        prob1 = BaseLegoFunction('Softmax', dict(name='prob1')).attach(n, [score1])
        prob2 = BaseLegoFunction('Softmax', dict(name='prob2')).attach(n, [score2])
        prob3 = BaseLegoFunction('Softmax', dict(name='prob3')).attach(n, [score3])
    elif control[netname] == 'VGG':
        n, score = vgg_layers(n, conf, learn_layers_vgg())
        prob = BaseLegoFunction('Softmax', dict(name='prob')).attach(n, [score])
    elif control[netname] == 'AlexNet':
        n, score = alexnet_layers(n, conf, learn_layers_alexnet())
        prob = BaseLegoFunction('Softmax', dict(name='prob')).attach(n, [score])
    else:
        raise

    if 'force_backward' in conf.keys():
        prefix = '\n\nforce_backward: true\n\n' + prefix

    return prefix + str(n.to_proto())


def nugu_train(conf, control, adv=None):
    prefix = ''

    n = caffe.NetSpec()

    n.data, n.label = L.Python(module='nugu.nugu_train_datalayer', layer='NuguDataLayer', ntop=2, param_str=str(dict(
        control=control,
        conf=conf,
        adv=adv,
    )))

    if 'ResNet' in control['net']:
        n, score = resnet_layers(n, True, control, conf, int(control['net'][6:]))
        loss = BaseLegoFunction('SoftmaxWithLoss', dict(name='loss')).attach(n, [score, n.label])
    elif control['net'] == 'GoogleNet':
        n, score1, score2, score3 = googlenet_layers(n, conf, learn_layers_googlenet(control))
        loss1 = BaseLegoFunction('SoftmaxWithLoss', dict(name='loss1', loss_weight=0.3)).attach(n, [score1, n.label])
        loss2 = BaseLegoFunction('SoftmaxWithLoss', dict(name='loss2', loss_weight=0.3)).attach(n, [score2, n.label])
        loss3 = BaseLegoFunction('SoftmaxWithLoss', dict(name='loss3', loss_weight=1)).attach(n, [score3, n.label])
    elif control['net'] == 'VGG':
        n, score = vgg_layers(n, conf, learn_layers_vgg(control))
        loss = BaseLegoFunction('SoftmaxWithLoss', dict(name='loss')).attach(n, [score, n.label])
    elif control['net'] == 'AlexNet':
        n, score = alexnet_layers(n, conf, learn_layers_alexnet(control))
        loss = BaseLegoFunction('SoftmaxWithLoss', dict(name='loss')).attach(n, [score, n.label])
    else:
        raise

    if 'force_backward' in conf.keys():
        prefix = '\n\nforce_backward: true\n\n' + prefix

    return prefix + str(n.to_proto())
