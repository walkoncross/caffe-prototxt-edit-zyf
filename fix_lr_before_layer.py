#!/usr/bin/env python
import sys
import os.path as osp
# import _init_paths # set caffe path if it's not in python's default lib path
from caffe_prototxt_editor import edit_net_prototxt


def fix_lr_before_layer(net, layer_name):
    fix_flag = True

    for l in net.layer:
        if l.name == layer_name:
            fix_flag = False

        if fix_flag:
            for param in l.param:
                param.lr_mult = 0
                param.decay_mult = 0


if __name__ == '__main__':
    input_fn = r'C:\zyf\github\caffe\models\bvlc_alexnet\train_val.prototxt'
    #input_fn = r'C:\zyf\github\caffe\models\bvlc_alexnet\deploy.prototxt'
    lr_before_layer_name = 'fc6'
    spl = osp.splitext(osp.basename(input_fn))
    output_fn = spl[0] + ('_fix_lr_before_%s' % lr_before_layer_name) + spl[1]

    if len(sys.argv) > 1:
        input_fn = sys.argv[1]
    if len(sys.argv) > 2:
        lr_before_layer_name = sys.argv[2]
    if len(sys.argv) > 3:
        output_fn = sys.argv[3]

    edit_net_prototxt(input_fn, output_fn, fix_lr_before_layer)
