__author__ = 'joon'

from caffe import layers as L
from caffe import params as P


def get_frozen_param():
    return [dict(lr_mult=0)] * 2


def get_learned_param():
    weight_param = dict(lr_mult=1, decay_mult=1)
    bias_param = dict(lr_mult=2, decay_mult=0)
    return [weight_param, bias_param]


def get_frozen_param_single():
    return dict(lr_mult=0)


def get_learned_param_single():
    weight_param = dict(lr_mult=1, decay_mult=1)
    return weight_param


def get_fixed_param():
    weight_param = dict(lr_mult=0, decay_mult=0)
    bias_param = dict(lr_mult=0, decay_mult=0)
    return [weight_param, bias_param]


def lrn(bottom, local_size, alpha, beta):
    norm = L.LRN(bottom, lrn_param=dict(local_size=local_size, alpha=alpha, beta=beta))
    return norm


def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1, dilation=1,
              param=get_learned_param(),
              weight_filler=dict(type='gaussian', std=0.01),
              bias_filler=dict(type='constant', value=0.1), engine=0, in_place=False):
    conv = L.Convolution(bottom,
                         param=param,
                         convolution_param=dict(
                             weight_filler=weight_filler,
                             bias_filler=bias_filler,
                             engine=engine,
                             num_output=nout, pad=pad, group=group,
                             kernel_size=ks, stride=stride, dilation=dilation,
                         ),
                         )
    return conv, L.ReLU(conv, in_place=in_place)


def conv_imprelu(bottom, ks, nout, stride=1, pad=0, group=1, dilation=1,
                 param=get_learned_param(),
                 weight_filler=dict(type='gaussian', std=0.01),
                 bias_filler=dict(type='constant', value=0.1), engine=0,
                 backtype='gradient'):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride, dilation=dilation,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler, engine=engine)

    relu = L.Python(conv, module='improper_relu_layers', layer='ImproperReLULayer', ntop=1,
                    param_str=str(dict(backtype=backtype)))
    return conv, relu


def conv(bottom, ks, nout, stride=1, pad=0, group=1, dilation=1,
         param=get_learned_param(),
         weight_filler=dict(type='gaussian', std=0.01),
         bias_filler=dict(type='constant', value=0.1)):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride, dilation=dilation,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
    return conv


def fc_relu(bottom, nout, param=get_learned_param(),
            weight_filler=dict(type='gaussian', std=0.005),
            bias_filler=dict(type='constant', value=0.1), in_place=False):
    fc = L.InnerProduct(bottom,
                        inner_product_param=dict(
                            num_output=nout,
                            weight_filler=weight_filler,
                            bias_filler=bias_filler),
                        param=param,
                        )
    return fc, L.ReLU(fc, in_place=in_place)


def fc(bottom, nout, param=get_learned_param(),
       weight_filler=dict(type='gaussian', std=0.005),
       bias_filler=dict(type='constant', value=0.1)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc


def fc_imprelu(bottom, nout, param=get_learned_param(),
               weight_filler=dict(type='gaussian', std=0.005),
               bias_filler=dict(type='constant', value=0.1),
               backtype='gradient'):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    relu = L.Python(fc, module='improper_relu_layers', layer='ImproperReLULayer', ntop=1,
                    param_str=str(dict(backtype=backtype)))
    return fc, relu


def back_imprelu(bottom0, bottom1, npred, backtype, param):
    if False:
        return L.Python(bottom0, bottom1, module='backward_improper_relu_layers', layer='BackwardImproperReLULayer',
                        ntop=1,
                        param_str=str(dict(backtype=backtype)))
    else:
        if backtype == 'gradient':
            backtype_int = 0
        elif backtype == 'deconv':
            backtype_int = 1
        elif backtype == 'guided':
            backtype_int = 2
        elif backtype == 'exper1':
            backtype_int = 3
        elif backtype == 'exper2':
            backtype_int = 4
        elif backtype == 'exper3':
            backtype_int = 5
        elif backtype == 'exper4':
            backtype_int = 6
        elif backtype == 'exper5':
            backtype_int = 7
        elif backtype == 'exper6':
            backtype_int = 8
        elif backtype == 'exper7':
            backtype_int = 9
        else:
            raise Exception('Choose correct backtype for backward improper relu layers')
        bottom1_nummatch = L.Python(bottom1, npred, module='util_python_layers', layer='ExpandNum',
                                    ntop=1, param_str=str(dict(target_num=20)))

    return L.BackImpReLU(bottom0, bottom1_nummatch, type=backtype_int, param=param)


def conv_bn_scale_relu(bottom, kernel_size, num_out, stride, pad, params):
    weight_filler = params[0]
    bias_filler = params[1]
    conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride, num_output=num_out,
                         pad=pad, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=weight_filler, bias_filler=bias_filler)
    bn_train = L.BatchNorm(conv, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0),
                                        dict(lr_mult=0, decay_mult=0)],
                           use_global_stats=False, in_place=True, include=dict(phase=0))
    bn_test = L.BatchNorm(conv, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0),
                                       dict(lr_mult=0, decay_mult=0)],
                          use_global_stats=True, in_place=True, include=dict(phase=1))
    scale = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)
    relu = L.ReLU(conv, in_place=True)

    return conv, bn_train, bn_test, scale, relu


def conv_bn_scale(bottom, kernel_size, num_out, stride, pad, params):
    weight_filler = params[0]
    bias_filler = params[1]
    conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride, num_output=num_out,
                         pad=pad, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=weight_filler, bias_filler=bias_filler)
    bn_train = L.BatchNorm(conv, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0),
                                        dict(lr_mult=0, decay_mult=0)],
                           use_global_stats=False, in_place=True, include=dict(phase=0))
    bn_test = L.BatchNorm(conv, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0),
                                       dict(lr_mult=0, decay_mult=0)],
                          use_global_stats=True, in_place=True, include=dict(phase=1))
    scale = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)

    return conv, bn_train, bn_test, scale


def unpool(bottom0, bottom1, ks, upsz, stride=1, method='max'):
    if method == 'max':
        if False:
            return L.Python(bottom0, bottom1, module='unpooling_layers', layer='UnmaxpoolLayer', ntop=1,
                            param_str=str(dict(ks=ks, stride=stride)))
        else:
            bottom1_switch = L.Python(bottom1, module='unpooling_layers', layer='ActToSwitch', ntop=1,
                                      param_str=str(dict(ks=ks, stride=stride)))
            return L.Unpooling(bottom0, bottom1_switch,
                               unpool_param=dict(unpool=0, kernel_size=ks, stride=stride,
                                                 unpool_size=upsz)), \
                   bottom1_switch
    else:
        raise Exception('unpooling currently only supports max pooling')


def max_pool(bottom, ks, stride=1, pad=0):
    return L.Pooling(bottom, pooling_param=dict(pool=P.Pooling.MAX, kernel_size=ks, stride=stride, pad=pad))


def scale(bottom, scale=1):
    return L.Power(bottom, power=1, scale=scale, shift=0)


def sigmoid(bottom):
    return L.Sigmoid(bottom)


def ptwisemult(bottom0, bottom1):
    return L.Eltwise(bottom0, bottom1, operation=0, stable_prod_grad=True)


def split(bottom):
    return L.Split(bottom)


def softmask(bottom, scale=1):
    dilseg_score_scaled = L.Power(bottom, scale=scale)
    return dilseg_score_scaled, L.Softmax(dilseg_score_scaled, axis=1)


def upsample(bottom, factor=1, channel=1):
    import numpy as np
    return L.Deconvolution(bottom,
                           convolution_param=dict(
                               weight_filler=dict(type='bilinear'),
                               kernel_size=2 * factor - factor % 2,
                               stride=factor,
                               num_output=channel,
                               group=channel,
                               pad=int(np.ceil((factor - 1.) / 2.)),
                               bias_term=False,
                           ),
                           param=dict(lr_mult=0, decay_mult=0)
                           )


def bw2rgb(bottom):
    return L.Concat(bottom, bottom, bottom, axis=1)


def selector(bottom, bottom_selector):
    return L.Filter(bottom, bottom_selector)


def select_by_masking(bottom0, bottom1):
    multoutput = ptwisemult(bottom0, bottom1)
    return multoutput, L.Convolution(multoutput, kernel_size=1, stride=1, num_output=1, param=get_fixed_param()[0],
                                     weight_filler=dict(
                                         type='constant',
                                         value=1,
                                     ),
                                     bias_term=False,
                                     )


def back_improper_prelu(bottom_dY, bottom_X, relu_type):
    if relu_type == 'gradient':
        prelu_filler = 1
    elif relu_type == 'guided':
        prelu_filler = 0
    else:
        prelu_filler = 1
    # todo: are the params _always_ learned?
    prelu = L.PReLU(bottom_dY, prelu_param=dict(filler=dict(value=prelu_filler)), channel_shared=1)
    backrelu = L.Python(prelu, bottom_X, module='backward_improper_relu_layers', layer='MultiplyWithSwitch', ntop=1)
    # switch_X = L.Threshold(bottom_X)
    # switch_X_fullchannel = L.Convolution(switch_X, num_output=20, kernel_size=1, param=dict(lr_mult=0, decay_mult=0),
    #                                      weight_filler=dict(value=1),
    #                                      bias_filler=dict(value=0))
    # backrelu = L.Eltwise(switch_X_fullchannel, prelu, operation=0, stable_prod_grad=True)
    return backrelu, prelu
