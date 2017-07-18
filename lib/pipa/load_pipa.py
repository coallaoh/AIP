__author__ = 'joon'

import os.path as osp


def load_pipa(piparoot):
    annofile = osp.join(piparoot, 'annotations/index.txt')
    fid = open(annofile, 'r')

    pipa = []

    for l in fid.readlines():
        l = l.strip()
        pipa.append(l.split())

    fid.close()

    return pipa


def load_splits(split, piparoot, load_numbers=False, ordered_ids=False):
    pipa = load_pipa(piparoot)
    print("loading splits..")
    splitroot = osp.join(piparoot,'annotations/splits')
    if type(split) == str:
        splitname = split
    elif type(split) == int:
        if split == 0:
            splitname = 'test_original'
        elif split == 2:
            splitname = 'test_album'
        elif split == 3:
            splitname = 'test_time'
        elif split == 5:
            splitname = 'test_day'
        elif split == 100:
            splitname = 'val_original'
        elif split == 102:
            splitname = 'val_album'
        elif split == 103:
            splitname = 'val_time'
        elif split == 105:
            splitname = 'val_day'
        else:
            raise

    if ordered_ids:
        splitfile = osp.join(splitroot, 'split_' + splitname + '.orderedids.txt')
    else:
        splitfile = osp.join(splitroot, 'split_' + splitname + '.txt')

    fid = open(splitfile, 'r')

    split0, split1 = [], []

    for l in fid.readlines():
        l = l.strip()
        l = l.split()
        if l[8] == '0':
            split0.append(l)
        elif l[8] == '1':
            split1.append(l)

    fid.close()

    pipa_ = [p[:6] for p in pipa]

    if load_numbers:
        split0_num, split1_num = [], []
        for split, split_num in zip([split0, split1], [split0_num, split1_num]):
            for l in split:
                num = pipa_.index(l[:6]) + 1
                split_num.append(num)
                # split_num.sort()
        print ("Done")
        return split0_num, split1_num
    else:
        print ("Done")
        return split0, split1
