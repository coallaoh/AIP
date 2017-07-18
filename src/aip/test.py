#!/usr/bin/env python

__author__ = 'joon'

import sys

sys.path.insert(0, 'src')

from imports.basic_modules import *
from imports.ResearchTools import *
from imports.import_caffe import *

from nugu.netdef import nugu_test

####
EXP_PHASE = 'aip_test'

conf = dict(
    liblinearpythonloc='/BS/joon_projects/work/liblinear/liblinear-2.1/python',
    piparoot='data',
    meanimageloc="data/ilsvrc_2012_mean.npy",
    pipacroploc="data",
    shuffle=True,
    vis=False,
    printoutput=True,
    save=True,
    overridecache=False,
    gpu=0,
    n=0,
    N=1,
)

control = dict(
    # net: network
    # Choose from {AlexNet, VGG, GoogleNet, ResNet152}
    net='GoogleNet',
    # i_method: AIP variant
    # Choose from {FGV(equiv to GA), FGS(equiv to BI), GAMAN, FGV-S(equiv to GA-S), FGV-S(equiv to BI-S)}
    i_method='GAMAN',
    # i_conf: AIP configuration
    # Should be in the format {alpha}_{iter}_{norm}
    # alpha: step size
    # iter: number of gradient ascent iterations
    # norm: L2 ball size -- at each iteration, if the perturbation is larger than the designated size, it will be projected.
    # Examples:
    # FGV with "1_100_1000" corresponds to GA (see paper) with 100 iterations and eps=1000.
    i_conf="1_100_1000",
    # i_process: AIP robustification
    # Should be in the format {processingtype1}{ensemblesize1}_{processingtype2}{ensemblesize2}_{...}
    # Each processingtype is chosen from {O,T,N,B,C}, meaning {No processing, Translation, Gaussian noise, Blurring, Cropping & resizing}
    # Examples:
    # "O1" indicates vanilla AIP.
    # "T5" indicates AIP robustified against translation. Gradients from five randomly translated samples are averaged.
    # "B1_C1" indicates AIP robustified against blurring and cropping. Gradients for each processing type are averaged.
    i_process='O1',
)


####

def config(control):
    # convnet training
    control['init'] = 'ImageNet'
    control['dataset'] = 'PIPA'
    control['datatype'] = 'TRAIN'
    control['part'] = 'head'
    control['baselr'] = 0.001
    control['lrpolicy'] = 'step50'
    control['resizing'] = 'rubber'

    # svm model
    control['code'] = 'h'
    control['C'] = 1
    control['norm'] = None
    control['svmtrainsplit'] = '0'

    # eval
    control['e_dataset'] = 'PIPA'
    control['e_datatype'] = 'VAL01'
    control['e_process'] = 'OTNBC_5'

    if control['net'] == 'AlexNet':
        control['batchsz'] = 50
        control['t_epoch'] = 100
    elif control['net'] == 'VGG':
        control['batchsz'] = 40
        control['t_epoch'] = 150
    elif control['net'] == 'GoogleNet':
        control['batchsz'] = 50
        control['t_epoch'] = 100
    elif control['net'] == 'ResNet152':
        control['batchsz'] = 10
        control['t_epoch'] = 25
    else:
        raise NotImplementedError

    ### svm model
    defaults_svm_model = dict(
        init='ImageNet',
        dataset='PIPA',
        datatype='TRAIN',
        lrpolicy='step50',
        baselr=0.001,
        batchsz=50,
        resizing='rubber',

        # SVM
        C=1,
        norm=None,
    )

    exclude_from_svm_model = [
        'part',
        'svmtrainsplit',
        'e_dataset',
        'e_datatype',
        'i_process',
        'i_method',
        'i_conf',
        'e_process',
    ]

    control_svm_model = control.copy()
    for ky in defaults_svm_model:
        if ky in control_svm_model.keys():
            if control_svm_model[ky] == defaults_svm_model[ky]:
                control_svm_model.pop(ky)

    for exc in exclude_from_svm_model:
        if exc in control_svm_model.keys():
            control_svm_model.pop(exc)

    if 'VAL' in control['e_datatype']:
        control_svm_model['t_datatype'] = 'VAL'
        control_svm_model['split'] = 100 + int(control['e_datatype'][3])
    else:
        raise NotImplementedError

    ### net model
    defaults_net_model = dict(
        dataset='PIPA',
        datatype='TRAIN',
        baselr=0.0001,
        resizing='rubber',
    )

    exclude_from_net_model = [
        'code',
        'C',
        'norm',
        'svmtrainsplit',
        'e_dataset',
        'e_datatype',
        'i_process',
        'i_method',
        'i_conf',
        'e_process',
        't_epoch',
    ]

    control_net_model = control.copy()
    for ky in defaults_net_model:
        if control_net_model[ky] == defaults_net_model[ky]:
            control_net_model.pop(ky)

    for exc in exclude_from_net_model:
        if exc in control_net_model.keys():
            control_net_model.pop(exc)

    ### token
    defaults_token = dict(
        # common
        init='ImageNet',
        dataset='PIPA',
        datatype='TRAIN',
        part='head',
        baselr=0.001,
        lrpolicy='step50',
        resizing='rubber',

        # for svm model
        code='h',
        C=1,
        norm=None,

        # ssam training set
        e_dataset='PIPA',
        e_datatype='VAL01',
        e_process=None,
    )

    control_token = control.copy()
    for ky in defaults_token:
        if ky in control_token.keys():
            if control_token[ky] == defaults_token[ky]:
                control_token.pop(ky)

    pprint.pprint(control)

    return control, control_svm_model, control_net_model, control_token


def format_conf(control, conf):
    conf['force_backward'] = True

    if control['net'] == 'AlexNet':
        conf['featname'] = 'fc7'
        conf['outname'] = 'fc8_'
        conf['inputsz'] = 227
        conf['baseepoch'] = 50
    elif control['net'] == 'VGG':
        conf['featname'] = 'fc7'
        conf['outname'] = 'fc8_'
        conf['inputsz'] = 224
        conf['baseepoch'] = 25
    elif control['net'] == 'GoogleNet':
        conf['featname'] = 'pool5_7x7_s1'
        conf['outname1'] = 'loss1_classifier_'
        conf['outname2'] = 'loss2_classifier_'
        conf['outname3'] = 'loss3_classifier_'
        conf['inputsz'] = 224
        conf['baseepoch'] = 25
    elif control['net'] == 'ResNet50':
        conf['featname'] = 'pool5'
        conf['outname'] = 'score'
        conf['inputsz'] = 224
        conf['baseepoch'] = 25
    elif control['net'] == 'ResNet101':
        conf['featname'] = 'pool5'
        conf['outname'] = 'score'
        conf['inputsz'] = 224
        conf['baseepoch'] = 5
    elif control['net'] == 'ResNet152':
        conf['featname'] = 'pool5'
        conf['outname'] = 'score'
        conf['inputsz'] = 224
        conf['baseepoch'] = 5
    else:
        raise NotImplementedError

    if control['datatype'] == 'TRAIN':
        conf['nout'] = 1409
        conf['ntrain'] = 29223
    else:
        raise NotImplementedError

    conf['svmtrainsplit'] = control['svmtrainsplit']

    if control['e_datatype'] == 'VAL00':
        conf['f_nout'] = 366
        conf['f_ntrain'] = 4820
    elif control['e_datatype'] == 'VAL01':
        conf['f_nout'] = 366
        conf['f_ntrain'] = 4820
    elif control['e_datatype'] == 'VAL20':
        conf['f_nout'] = 366
        conf['f_ntrain'] = 4859
    elif control['e_datatype'] == 'VAL21':
        conf['f_nout'] = 366
        conf['f_ntrain'] = 4783
    elif control['e_datatype'] == 'VAL30':
        conf['f_nout'] = 366
        conf['f_ntrain'] = 4818
    elif control['e_datatype'] == 'VAL31':
        conf['f_nout'] = 366
        conf['f_ntrain'] = 4824
    elif control['e_datatype'] == 'VAL50':
        conf['f_nout'] = 65
        conf['f_ntrain'] = 1076
    elif control['e_datatype'] == 'VAL51':
        conf['f_nout'] = 65
        conf['f_ntrain'] = 1076
    elif control['e_datatype'] == 'TEST00':
        conf['f_nout'] = 581
        conf['f_ntrain'] = 6443
    elif control['e_datatype'] == 'TEST01':
        conf['f_nout'] = 581
        conf['f_ntrain'] = 6443
    elif control['e_datatype'] == 'TEST20':
        conf['f_nout'] = 581
        conf['f_ntrain'] = 6497
    elif control['e_datatype'] == 'TEST21':
        conf['f_nout'] = 581
        conf['f_ntrain'] = 6389
    elif control['e_datatype'] == 'TEST30':
        conf['f_nout'] = 581
        conf['f_ntrain'] = 6441
    elif control['e_datatype'] == 'TEST31':
        conf['f_nout'] = 581
        conf['f_ntrain'] = 6445
    elif control['e_datatype'] == 'TEST50':
        conf['f_nout'] = 199
        conf['f_ntrain'] = 2484
    elif control['e_datatype'] == 'TEST51':
        conf['f_nout'] = 199
        conf['f_ntrain'] = 2485
    else:
        raise NotImplementedError

    if 'VAL' in control['e_datatype']:
        conf['split'] = int(control['e_datatype'][3]) + 100
    elif 'TEST' in control['e_datatype']:
        conf['split'] = int(control['e_datatype'][4])
    else:
        raise NotImplementedError

    conf['split01'] = int(control['e_datatype'][-1])

    if control['net'] == 'AlexNet':
        conf['featdim'] = 4096
    elif control['net'] == 'VGG':
        conf['featdim'] = 4096
    elif control['net'] == 'GoogleNet':
        conf['featdim'] = 1024
    elif 'ResNet' in control['net']:
        conf['featdim'] = 2048
    else:
        raise NotImplementedError

    conf['inputname'] = 'data'

    return conf


def translate_image(im, h, w):
    new_im = np.zeros_like(im).astype(np.float)
    if h > 0:
        if w > 0:
            new_im[h:, w:] = im[:-h, :-w]
        elif w < 0:
            new_im[h:, :w] = im[:-h, -w:]
        else:
            new_im[h:, :] = im[:-h, :]
    elif h < 0:
        if w > 0:
            new_im[:h, w:] = im[-h:, :-w]
        elif w < 0:
            new_im[:h, :w] = im[-h:, -w:]
        else:
            new_im[:h, :] = im[-h:, :]
    else:
        if w > 0:
            new_im[:, w:] = im[:, :-w]
        elif w < 0:
            new_im[:, :w] = im[:, -w:]
        else:
            new_im[:, :] = im[:, :]
    return new_im


def compute_aip_main(input, l, f, Df, itr, stepsize, maxnorm, fgv_method, process, init=None, clip_limits=None,
                     bw_flag=False):
    def deproc_jitter(neut_type, dx_proc, conf_proc):
        if neut_type == 'O':
            return dx_proc.copy()
        elif neut_type == 'T':
            if conf_proc['zero']:
                return np.zeros_like(dx_proc)
            else:
                return translate_image(dx_proc.transpose([1, 2, 0]), -conf_proc['h'], -conf_proc['w']).transpose(
                    [2, 0, 1])
        elif neut_type == 'N':
            return dx_proc.copy()
        elif neut_type == 'B':
            return cv2.GaussianBlur(dx_proc.transpose([1, 2, 0]).astype(np.float),
                                    (conf_proc['sigma'], conf_proc['sigma']),
                                    0).transpose([2, 0, 1])
        elif neut_type == 'b':
            return cv2.GaussianBlur(dx_proc.transpose([1, 2, 0]).astype(np.float),
                                    (conf_proc['sigma'], conf_proc['sigma']),
                                    0).transpose([2, 0, 1])
        elif neut_type == 'C':
            if conf_proc['zero']:
                return np.zeros_like(dx_proc)
            else:
                dx_proc = nd.zoom(dx_proc, 1. / conf_proc['ratio'], order=1)
                dx = np.zeros(conf_proc['origshape'], dtype=np.float32)
                dx[:, conf_proc['x0']:conf_proc['x1'], conf_proc['y0']:conf_proc['y1']] = dx_proc
                return dx
        else:
            raise NotImplementedError

    def compute_jittered(neut_type, x_t):
        if neut_type == 'O':
            return x_t, dict()
        elif neut_type == 'T':
            x_t_proc, conf_proc = random_translation(x_t.transpose([1, 2, 0]), return_coords=True)
            x_t_proc = x_t_proc.transpose([2, 0, 1])
            conf_proc['zero'] = False
            if (x_t_proc.shape[1] < 5) or (x_t_proc.shape[2] < 5):
                conf_proc['zero'] = True
                return np.zeros_like(x_t), conf_proc
            else:
                return x_t_proc, conf_proc
        elif neut_type == 'N':
            x_t_proc = x_t.copy().astype(np.float)
            noise = np.random.normal(scale=10, size=x_t.shape)
            x_t_proc += noise
            return x_t_proc, dict()
        elif neut_type == 'B':
            sigma = np.random.choice(range(1, 5), 1)[0] * 2 + 1
            x_t_proc = cv2.GaussianBlur(x_t.astype(np.float).transpose([1, 2, 0]), (sigma, sigma), 0).transpose(
                [2, 0, 1])
            return x_t_proc, dict(sigma=sigma)
        elif neut_type == 'b':
            sigma = np.random.choice(range(1, 5), 1)[0] * 2 + 1
            x_t_proc = cv2.GaussianBlur(x_t.astype(np.float).transpose([1, 2, 0]), (sigma, sigma), 0).transpose(
                [2, 0, 1])
            return x_t_proc, dict(sigma=sigma)
        elif neut_type == 'C':
            x_t_proc, conf_proc = random_crop(x_t.transpose([1, 2, 0]), return_coords=True)
            conf_proc['origshape'] = x_t.shape
            x_t_proc = x_t_proc.transpose([2, 0, 1])
            conf_proc['zero'] = False
            if (x_t_proc.shape[1] < 5) or (x_t_proc.shape[2] < 5):
                conf_proc['zero'] = True
                return np.zeros_like(x_t), conf_proc
            else:
                ratio = np.array(x_t.shape) / np.array(x_t_proc.shape).astype(np.float)
                x_t_proc = nd.zoom(x_t_proc, ratio, order=1)
                conf_proc['ratio'] = ratio
                return x_t_proc, conf_proc
        else:
            raise NotImplementedError

    n = 0
    if init is None:
        res = np.zeros_like(input)
    else:
        res = init.copy()

    def stopping_condition(i, r):
        return i < itr

    process_parsed = process.split('_')

    while stopping_condition(n, res):
        n += 1
        x_t = input + res

        dx_deproc_stacked = []
        for neut in process_parsed:
            neut_type = neut[0]
            nbat = int(neut[1:])
            for bat_id in range(nbat):
                im_, conf_ = compute_jittered(neut_type, x_t)
                dx_proc = Df(im_, l, fgv_method)
                dx_ = deproc_jitter(neut_type, dx_proc, conf_)
                dx_deproc_stacked.append(dx_.reshape(1, *x_t.shape))
        dx = np.concatenate(dx_deproc_stacked, axis=0).mean(0)

        if fgv_method in ['FGV-S', 'FGV']:
            dxx = dx * 10000 * stepsize
        elif fgv_method in ['GAMAN']:
            dxx = dx * 5000 * stepsize
        elif fgv_method in ['FGS-S', 'FGS']:
            dxx = np.sign(dx) * 0.1 * stepsize
        else:
            raise NotImplementedError

        if bw_flag:
            mean = dxx.mean(0)
            dxx[0] = mean
            dxx[1] = mean
            dxx[2] = mean

        res -= dxx

        if clip_limits is not None:
            lower_bound = np.tile(clip_limits[0].reshape(3, 1, 1), [1, input.shape[1], input.shape[2]])
            upper_bound = np.tile(clip_limits[1].reshape(3, 1, 1), [1, input.shape[1], input.shape[2]])
            I_too_low = (input + res) <= lower_bound
            I_too_high = (input + res) >= upper_bound
            res[I_too_low] = (lower_bound - input)[I_too_low]
            res[I_too_high] = (upper_bound - input)[I_too_high]
            assert ((input + res) >= lower_bound - 1e-4).all()
            assert ((input + res) <= upper_bound + 1e-4).all()

        if maxnorm != 0:
            res = proj_lp(res, maxnorm, 2)

    return res, n


def apply_neutralisation(im, neut_proc, imdomain=False):
    im_out = []
    neut_profile = []

    neut_types, how_many = neut_proc.split('_')
    how_many = int(how_many)

    # assert (neut_types == 'OTNBC')

    def transform_imdomain(im_proc):
        if imdomain:
            im_proc = np.maximum(im_proc, 0)
            im_proc = np.minimum(im_proc, 255)
            im_proc = im_proc.astype(np.uint8)
        return im_proc

    for _ in range(how_many):
        for neut_type in neut_types:
            if neut_type == 'O':
                im_out.append(im.copy())
            elif neut_type == 'T':
                im_proc = random_translation(im)
                if (im_proc.shape[0] < 5) or (im_proc.shape[1] < 5):
                    im_proc = np.zeros((10, 10, 3), dtype=np.float)
                im_out.append(transform_imdomain(im_proc))
            elif neut_type == 'N':
                im_proc = im.copy().astype(np.float)
                noise = np.random.normal(scale=10, size=im.shape)
                im_proc += noise
                im_out.append(transform_imdomain(im_proc))
            elif neut_type == 'B':
                sigma = np.random.choice(range(5), 1)[0] * 2 + 1
                im_proc = cv2.GaussianBlur(im.astype(np.float), (sigma, sigma), 0)
                im_out.append(transform_imdomain(im_proc))
            elif neut_type == 'C':
                im_proc = random_crop(im)
                if (im_proc.shape[0] < 5) or (im_proc.shape[1] < 5):
                    im_proc = np.zeros((10, 10, 3), dtype=np.float)
                im_proc = nd.zoom(im_proc, np.array(im.shape) / np.array(im_proc.shape).astype(np.float), order=1)
                im_out.append(transform_imdomain(im_proc))
            else:
                raise NotImplementedError
            neut_profile.append(neut_type)

    return im_out, neut_profile


def test_up(net, svm_w, testlist, labellist, conf, control, outdir):
    mu = np.load(conf['meanimageloc']).mean(1).mean(1)
    transformer = set_preprocessor_without_net(
        [1, 3, conf['inputsz'], conf['inputsz']], mean_image=mu)
    transformer_pert = set_preprocessor_without_net([1, 3, conf['inputsz'], conf['inputsz']],
                                                    mean_image=np.zeros_like(mu))

    num_images = len(testlist)
    print('%d images for training perturbation' % num_images)

    i_conf = control['i_conf'].split('_')
    if control['i_method'] in ['FGV-S', 'FGS-S', 'GAMAN', 'FGV', 'FGS']:
        stepsize = float(i_conf[0])
        itr = float(i_conf[1])
        maxnorm = float(i_conf[2])
    else:
        raise NotImplementedError

    def f(x, return_label=True, inputname=conf['inputname']):
        net.blobs[inputname].data.flat = x.flat
        net.forward()
        feat = net.blobs[conf['featname']].data.reshape(-1)
        out = np.dot(svm_w.T, feat)
        if return_label:
            return out.argmax()
        else:
            return out

    def Df(input, l, fgv_method, featname=conf['featname'], inputname=conf['inputname']):
        net.blobs[inputname].data.flat = input.flat
        net.forward()

        def compute_dfdl():
            dfdl = np.zeros(svm_w.shape[1], dtype=np.float32)
            if fgv_method in ['FGV-S', 'FGS-S']:
                dfdl[l] = 1
            elif fgv_method in ['GAMAN']:
                out = f(input, return_label=False)
                labs = np.argsort(out)[::-1]
                l2 = labs[0] if labs[0] != l else labs[1]
                dfdl[l] = 1
                dfdl[l2] = -1
            elif fgv_method in ['FGV', 'FGS']:
                dfdl[l] = 1
                sco = f(input, return_label=False)
                sm = Jsoftmax(sco, axis=0)
                dfdl -= sm
            else:
                raise NotImplementedError
            return dfdl

        dfdl = compute_dfdl()
        dfdfeat = np.dot(svm_w, dfdl)
        net.blobs[featname].diff.flat = dfdfeat.flat
        net.backward(start=featname)
        dfdx = net.blobs[inputname].diff[0].copy()
        return dfdx

    def compute_aip(input, l, init, bw_flag=False):
        if control['i_method'] in ['FGV-S', 'FGS-S', 'GAMAN', 'FGV', 'FGS']:
            res, final_itr = compute_aip_main(input, l, f, Df, itr, stepsize, maxnorm,
                                              control['i_method'],
                                              process=control['i_process'],
                                              init=init,
                                              clip_limits=[-mu, 255 - mu],
                                              bw_flag=bw_flag)
            print("TOOK %d iter" % final_itr)
            return res

        else:
            raise NotImplementedError

    def compute_neut_labels(s_pred_neut, neut_profile, process):

        neut_types, how_many = process.split('_')
        how_many = int(how_many)

        l__ = {}

        for neut_type in neut_types:
            inds = np.where(np.array(neut_profile) == neut_type)[0]
            ensemble = s_pred_neut[inds].cumsum(axis=0)
            l_en = ensemble.argmax(axis=1)
            l__[neut_type] = l_en

        l__['E'] = s_pred_neut.cumsum(axis=0)[0::len(neut_types)].argmax(axis=1)

        return l__

    def nugu_train_improcess_test(im, conf, return_bw_flag=False):
        bw_flag = False
        if len(im.shape) == 2:
            im = bw_to_rgb(im)
            bw_flag = True
        elif im.shape[2] == 1:
            im = bw_to_rgb(im)
            bw_flag = True
        else:
            assert (len(im.shape) == 3)
            assert (im.shape[2] == 3)

        if (im.shape[0] <= 1) or (im.shape[1] <= 1):
            im = np.zeros((conf['inputsz'], conf['inputsz'], 3), dtype=np.float32)

        imshape_original = im.shape[:2]

        if control['resizing'] == 'mirror':
            im = scipy.misc.imresize(im, float(conf['inputsz']) / max(imshape_original))

            imshape = im.shape[:2]
            margin = [(conf['inputsz'] - imshape[0]) // 2, (conf['inputsz'] - imshape[1]) // 2]
            im = cv2.copyMakeBorder(im, margin[0], conf['inputsz'] - imshape[0] - margin[0],
                                    margin[1], conf['inputsz'] - imshape[1] - margin[1],
                                    cv2.BORDER_REFLECT_101)

        elif control['resizing'] == 'rubber':
            im = scipy.misc.imresize(im, (conf['inputsz'], conf['inputsz']))

        else:
            raise NotImplementedError

        assert (im.shape[0] == im.shape[1] == conf['inputsz'])

        if return_bw_flag:
            return im, bw_flag
        else:
            return im

    def compute_indiv():
        fooling_stats = []
        stime = time.time()
        n_skipped = 0
        print("Saving to %s" % outdir)
        print("Saving to %s" % outdir)
        print("Saving to %s" % outdir)
        for idx in range(num_images):
            time_taken = time.time() - stime
            amortised_time = time_taken / ((idx - n_skipped) + 1e-10)
            print ('Computing fooling rate.. set %d / %d   image %d / %d' % (
                conf['n'] + 1, conf['N'], idx + 1, num_images))
            print ('ETA in %2.2f sec' % ((num_images - idx) * amortised_time))
            im_id = testlist[idx]

            resfile = osp.join(outdir, 'fooling_stats_' + str(im_id) + '.pkl')

            if osp.isfile(resfile):
                fooling_stats_i = load_from_cache(resfile)
                fooling_stats.append(fooling_stats_i.copy())
                print("Skipping %s" % im_id)
                n_skipped += 1
                continue

            l_gt = labellist[idx]
            imname = osp.join(conf['pipacroploc'], control['part'] + '_crop', str(im_id) + '.jpg')
            im_original = load_image_PIL(imname)
            im = im_original.copy()
            im, bw_flag = nugu_train_improcess_test(im, conf, return_bw_flag=True)
            input = transformer.preprocess('data', im)
            l_pred = f(input)

            dv = compute_aip(input, l_gt, init=None, bw_flag=bw_flag)
            pert_imdomain = transformer_pert.deprocess('data', dv).copy()

            def combine_im_pert(im, pert):
                ratio = np.array(im.shape).astype(np.float) / np.array(pert.shape).astype(np.float)
                pert = nd.zoom(pert, ratio)
                imp = im + pert
                imp = np.maximum(imp, 0)
                imp = np.minimum(imp, 255)
                imp = imp.astype(np.uint8)
                return imp

            imr_o = combine_im_pert(im_original, pert_imdomain)
            l_pred_fool = f(input + dv)
            L2 = np.linalg.norm(dv.reshape(-1), 2)

            imr_o_tmp = nugu_train_improcess_test(imr_o, conf)
            input_r = transformer.preprocess('data', imr_o_tmp)
            l_pred_fool_postproc = f(input_r)

            imr_os, neut_profile = apply_neutralisation(imr_o, imdomain=True, neut_proc=control['e_process'])
            ims, _ = apply_neutralisation(im_original, imdomain=True, neut_proc=control['e_process'])

            s_pred_fool_postproc_neut = []
            s_pred_neut = []

            for imr_o_, im_ in zip(imr_os, ims):
                imr_o_ = nugu_train_improcess_test(imr_o_, conf)
                im_ = nugu_train_improcess_test(im_, conf)
                input_r_ = transformer.preprocess('data', imr_o_)
                input_ = transformer.preprocess('data', im_)
                s_pred_fool_postproc_neut.append(f(input_r_, return_label=False).reshape((1, -1)))
                s_pred_neut.append(f(input_, return_label=False).reshape((1, -1)))

            s_pred_fool_postproc_neut = np.concatenate(s_pred_fool_postproc_neut, axis=0)
            s_pred_neut = np.concatenate(s_pred_neut, axis=0)

            l_pred_neut = compute_neut_labels(s_pred_neut, neut_profile, control['e_process'])
            l_pred_fool_postproc_neut = compute_neut_labels(s_pred_fool_postproc_neut, neut_profile,
                                                            control['e_process'])

            if conf['vis']:

                def visualise_adversarial_perturbation(imo, l_gt, l_pred, pert, imr, l_pred_fool, l_pred_fool_postproc,
                                                       L2):
                    fig = plt.figure(0)
                    # fig.suptitle('I: {}'.format(idx))
                    ax = fig.add_subplot(1, 3, 1)
                    ax.set_title('Original image\nGT: %d, PRED: %d' % (l_gt, l_pred))
                    pim(imo)
                    ax = fig.add_subplot(1, 3, 2)
                    ax.set_title('Adversarial\nnoise\nL2=%d' % (L2))
                    if pert is not None:
                        pim((
                                # 255 * (pert - pert.min()) / (pert.max() - pert.min())
                                pert + 127
                            ).astype(np.uint8))
                    ax = fig.add_subplot(1, 3, 3)
                    ax.set_title('Perturbed image\nFOOL: %d, FOOL&PROC: %d' % (l_pred_fool, l_pred_fool_postproc))
                    pim(imr)
                    for iii in range(3):
                        fig.axes[iii].get_xaxis().set_visible(False)
                        fig.axes[iii].get_yaxis().set_visible(False)
                    plt.pause(0.1)
                    return

                visualise_adversarial_perturbation(imo=im_original, l_gt=l_gt, l_pred=l_pred, pert=pert_imdomain,
                                                   imr=imr_o, l_pred_fool=l_pred_fool,
                                                   l_pred_fool_postproc=l_pred_fool_postproc, L2=L2)
            if conf['printoutput']:
                print("GT image label ........................ %d" % l_gt)
                print("Clean image prediction ................ %d" % l_pred)
                for neut_type in l_pred_neut.keys():
                    print("Clean image + R's defense type %s ...... %d" % (neut_type, l_pred_neut[neut_type][-1]))
                print("Perturbed image prediction ............ %d" % l_pred_fool)
                print("Pert + Proc (resize & quantise) ....... %d" % l_pred_fool_postproc)
                for neut_type in l_pred_fool_postproc_neut.keys():
                    print("Pert + Proc + R's defense type %s ...... %d" % (neut_type, l_pred_fool_postproc_neut[neut_type][-1]))
                print("||r||_2 = %2.2f" % L2)

            fooling_stats_i = dict(
                l_gt=l_gt,
                l_pred=l_pred,
                l_pred_neut=l_pred_neut,
                l_pred_fool=l_pred_fool,
                l_pred_fool_postproc=l_pred_fool_postproc,
                l_pred_fool_postproc_neut=l_pred_fool_postproc_neut,
                norm=L2,
            )
            fooling_stats.append(fooling_stats_i.copy())

            save_to_cache(fooling_stats_i, resfile)

        return fooling_stats

    fooling_stats = compute_indiv()

    return fooling_stats


def show_stat(fooling_stats):
    l_gt = np.array([r['l_gt'] for r in fooling_stats])
    l_pred = np.array([r['l_pred'] for r in fooling_stats])
    l_pred_fool = np.array([r['l_pred_fool'] for r in fooling_stats])
    l_pred_fool_postproc = np.array([r['l_pred_fool_postproc'] for r in fooling_stats])
    r_norm = np.array([r['norm'] for r in fooling_stats]).mean()

    acc = (l_gt == l_pred).sum() / float(len(l_gt)) * 100
    acc_fool = (l_gt == l_pred_fool).sum() / float(len(l_gt)) * 100
    acc_fool_postproc = (l_gt == l_pred_fool_postproc).sum() / float(len(l_gt)) * 100
    fooling_rate = ((l_gt == l_pred) & (l_gt != l_pred_fool)).sum() / float((l_gt == l_pred).sum()) * 100

    print('Norm             = %2.2f' % (r_norm))
    print('Accuracy         = %2.2f' % (acc))
    print('+ adversary      = %2.2f' % (acc_fool))
    print('+ improc         = %2.2f' % (acc_fool_postproc))
    print('New fooling rate = %2.2f' % (fooling_rate))

    neut_types, how_many = control['e_process'].split('_')
    how_many = int(how_many)
    presentation_neut = {}

    if control['e_process'] is not None:
        print('========= Neutralisation =========')
        for idx_neut, neut_type in enumerate(neut_types + 'E'):
            if neut_type == 'O':
                continue
            elif neut_type == 'T':
                neut_name = 'Translate'
            elif neut_type == 'N':
                neut_name = 'Noise'
            elif neut_type == 'B':
                neut_name = 'Blur'
            elif neut_type == 'C':
                neut_name = 'Crop'
            elif neut_type == 'E':
                neut_name = 'Altogether'
            else:
                raise NotImplementedError

            l_pred_neut = np.array([r['l_pred_neut'][neut_type] for r in fooling_stats])
            acc_neut = (np.tile(l_gt.reshape((-1, 1)), [1, how_many]) == l_pred_neut).sum(0) / float(len(l_gt)) * 100
            l_pred_fool_postproc_neut = np.array([r['l_pred_fool_postproc_neut'][neut_type] for r in fooling_stats])
            acc_fool_postproc_neut = (np.tile(l_gt.reshape((-1, 1)), [1, how_many]) == l_pred_fool_postproc_neut).sum(
                0) / float(len(l_gt)) * 100

            for idx_en in range(how_many):
                print('%10s accuracy ensemble %2d (no fool)     = %2.2f' % (neut_name, idx_en + 1, acc_neut[idx_en]))
            for idx_en in range(how_many):
                print('%10s accuracy ensemble %2d (fool)        = %2.2f' % (
                    neut_name, idx_en + 1, acc_fool_postproc_neut[idx_en]))

            presentation_neut[neut_type] = acc_fool_postproc_neut[-1]

    # printout = [r_norm, acc_fool, acc_fool_postproc] + acc_postproc_neut
    # print("a={}".format(printout))
    if len(control['e_process']) > 3:
        print("%2.1f & %2.1f & %2.1f & %2.1f & %2.1f & %2.1f" % (
            acc_fool_postproc, presentation_neut['T'], presentation_neut['N'], presentation_neut['B'],
            presentation_neut['C'], presentation_neut['E']))

    return


def run_adv(control, control_svm_model, control_net_model, control_token, conf, EXP_PHASE):
    # outfile
    outdir = osp.join('cache', EXP_PHASE, create_token(control_token))
    mkdir_if_missing(outdir)
    print('saving to: {}'.format(outdir))

    def crawl_net(control_net_model, control, control_token, conf, GPU):
        # trained model
        learnedmodel_dir = osp.join('cache', 'nugu_train', create_token(control_net_model))
        mkdir_if_missing(learnedmodel_dir)
        learnedmodel = osp.join(learnedmodel_dir,
                                '_iter_' + str((conf['ntrain'] * conf['baseepoch'] / control['batchsz']) * (
                                    control['t_epoch'] / conf['baseepoch'])) + '.caffemodel')

        # prototxt
        protodir = osp.join('models', EXP_PHASE, create_token(control_token))
        mkdir_if_missing(protodir)
        testproto = osp.join(protodir, 'test.prototxt')

        def write_proto(testproto, conf, control):
            f = open(testproto, 'w')
            f.write(nugu_test(conf, control))
            f.close()
            return

        write_proto(testproto, conf, control)

        # init
        caffe.set_mode_gpu()
        caffe.set_device(GPU)

        net = caffe.Net(testproto, learnedmodel, caffe.TEST)
        disp_net(net)

        return net

    def crawl_svm(control_svm_model, conf):
        # load svm model
        modelfile = osp.join('cache', 'nugu_svm', create_token(control_svm_model), 'model' + conf['svmtrainsplit'])

        sys.path.insert(0, conf['liblinearpythonloc'])
        from liblinearutil import load_model as svm_load_model
        import liblinear

        svm_model = svm_load_model(modelfile)

        def crawl_svm_weights(svm_model, featdim, labeldim):
            print("    Crawling SVM weights from the model")
            weights = np.zeros((featdim, labeldim), dtype=np.float32)
            for ii in range(1, 1 + featdim):
                for jj in range(labeldim):
                    weights[ii - 1, jj] = liblinear.liblinear.get_decfun_coef(svm_model, ii, jj)
            return weights

        svm_w = crawl_svm_weights(svm_model, conf['featdim'], conf['f_nout'])
        return svm_w

    net = crawl_net(control_net_model, control, control_token, conf, conf['gpu'])
    svm_w = crawl_svm(control_svm_model, conf)

    def load_testset(conf):
        # extraction
        from pipa.load_pipa import load_splits

        split0, split1 = load_splits(conf['split'], piparoot=conf['piparoot'], load_numbers=True, ordered_ids=True)
        split0, split1 = np.array(split0), np.array(split1)
        split0_f, split1_f = load_splits(conf['split'], piparoot=conf['piparoot'], load_numbers=False, ordered_ids=True)
        id0 = np.array([l[9] for l in split0_f]).astype(np.int)
        id1 = np.array([l[9] for l in split1_f]).astype(np.int)

        if conf['split01'] == 0:
            test_list = split0.copy()
            labellist = id0.copy()
        else:
            test_list = split1.copy()
            labellist = id1.copy()

        if conf['shuffle']:
            np.random.seed(2512)
            tmpv = range(len(test_list))
            np.random.shuffle(tmpv)
            test_list = test_list[tmpv]
            labellist = labellist[tmpv]

        return test_list, labellist

    testlist, labellist = load_testset(conf)

    def get_subset(testlist, labellist, n, N):
        totlen = len(testlist)
        return testlist[np.round(totlen * n / N):np.round(totlen * (n + 1) / N)], \
               labellist[np.round(totlen * n / N):np.round(totlen * (n + 1) / N)]

    testlist, labellist = get_subset(testlist, labellist, conf['n'], conf['N'])

    fooling_stats = test_up(net, svm_w, testlist, labellist, conf, control, outdir)
    show_stat(fooling_stats)

    return


if __name__ == "__main__":
    control, control_svm_model, control_net_model, control_token = config(control)

    conf = format_conf(control, conf)

    run_adv(control, control_svm_model, control_net_model, control_token, conf, EXP_PHASE)
