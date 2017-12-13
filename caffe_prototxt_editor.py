#!/usr/bin/env python

# import _init_paths # set caffe path if it's not in python's default lib path

from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtfmt


def print_net_layers(net):
    print 'net.name: ', net.name
    print 'Layers: '
    for i, l in enumerate(net.layer):
        print '-->%2d layer: %s' % (i + 1, l.name)


def load_net_from_prototxt(input_fn):
    print 'Load net from: ', input_fn
    net = caffe_pb2.NetParameter()

    with open(input_fn) as fp:
        s = fp.read()
        txtfmt.Merge(s, net)

    print 'Loading finished'
    print_net_layers(net)
    return net


def save_net_to_prototxt(net, output_fn):
    print 'Saving net params into ', output_fn
    with open(output_fn, 'w') as fp:
        fp.write(str(net))
    print 'Saving finished'


# example for editing net params
def edit_net_params_ex(net, params=None):
    net.name = 'my new net'
    layer_names = [l.name for l in net.layer]
    idx = layer_names.index('fc6')
    l = net.layer[idx]
    l.param[0].lr_mult = 1.3


def edit_net_prototxt(input_fn, output_fn, edit_func=edit_net_params_ex, edit_func_params=None):
    net = load_net_from_prototxt(input_fn)

    edit_func(net, edit_func_params)
    save_net_to_prototxt(net, output_fn)


if __name__ == '__main__':
    input_fn = r'C:\zyf\github\caffe\models\bvlc_alexnet\train_val.prototxt'
    #input_fn = r'C:\zyf\github\caffe\models\bvlc_alexnet\deploy.prototxt'
    output_fn = './my_new_net.prototxt'

    edit_net_prototxt(input_fn, output_fn)
