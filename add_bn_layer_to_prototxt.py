#!/usr/bin/env python
# # -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 02:43:55 2017

@author: zhaoy
"""
import sys
import os.path as osp


# ADD_LR_PARAM = True


def split_lines_with_bracket(line):
    spl = []
    cnt0 = 0
    cnt1 = 0

    pos0 = 0

    line = line.strip()

    for i, ch in enumerate(line):
        if ch == '{':
            spl.append(line[pos0:i + 1])
#            spl.append('{')
            pos0 = i + 1
            cnt0 += 1
        elif ch == '}':
            spl.append(line[pos0:i])
            spl.append('}')
            pos0 = i + 1
            cnt1 += 1

    if line[pos0:]:
        spl.append(line[pos0:])

    return spl, cnt0, cnt1


lr_param_for_weight = '''
param {
    lr_mult: 1
    decay_mult: 1
  }
'''

lr_param_for_bias = '''
param {
    lr_mult: 2
    decay_mult: 0
}
'''

lr_param_for_weight_nodecay = '''
param {
    lr_mult: 1
  }
'''

lr_param_for_bias_nodecay = '''
param {
    lr_mult: 2
}
'''

lr_param_fixed = '''
param {
    lr_mult: 0
    decay_mult: 0
}
'''

lr_param_fixed_nodecay = '''
param {
    lr_mult: 0
    decay_mult: 0
}
'''


def add_lr_param(lines, param):
    for sp in param.split('\n'):
        lines.append(sp.strip())


bn_layer_template = '''
layer {
  bottom: "<blob_name>"
  top: "<blob_name>"
  name: "<layer_name>_bn"
  type: "BatchNorm"
  batch_norm_param {

  }
}
'''

scale_layer_template = '''
layer {
  bottom: "<blob_name>"
  top: "<blob_name>"
  name: "<layer_name>_scale"
  type: "Scale"
  scale_param {
    bias_term: true
  }
}
'''

relu_layer_template = '''
layer {
  bottom: "<blob_name>"
  top: "<blob_name>"
  name: "<layer_name>_relu"
  type: "ReLU"
}
'''

prelu_layer_template = '''
layer {
  bottom: "<blob_name>"
  top: "<blob_name>"
  name: "<layer_name>_prelu"
  type: "PReLU"
}
'''


def append_bn_layer(layer_name, blob_name):
    bn_layer = bn_layer_template.replace('<layer_name>', layer_name)
    bn_layer = bn_layer.replace('<blob_name>', blob_name)

    return bn_layer


def append_scale_layer(layer_name, blob_name):
    scale_layer = scale_layer_template.replace('<layer_name>', layer_name)
    scale_layer = scale_layer.replace('<blob_name>', blob_name)

    return scale_layer


def append_relu_layer(layer_name, blob_name):
    relu_layer = relu_layer_template.replace('<layer_name>', layer_name)
    relu_layer = relu_layer.replace('<blob_name>', blob_name)

    return relu_layer


def append_prelu_layer(layer_name, blob_name):
    prelu_layer = prelu_layer_template.replace('<layer_name>', layer_name)
    prelu_layer = prelu_layer.replace('<blob_name>', blob_name)

    return prelu_layer


skip_relu_conv_layers = []
for i in range(1, 6):
    for j in range(1, 17):
        skip_layer_name = 'conv%d_%d' % (i, j * 2 + 1)
        skip_relu_conv_layers.append(skip_layer_name)


def write_section(fp, lines, cnt):
    if not lines:
        return
    else:
        tab_cnt = 0
        new_lines = []

#        is_layer = False
        layer_type = ''
        layer_name = ''
        has_lr_param = False
        has_bias_term = False

        bottom_blobs = []
        top_blobs = []

        for i in range(len(lines)):
            line = lines[i]
            # # rename the Net
            # if cnt == 0 and line.startswith('name:'):
            #     name = line.split(':')[1].strip()
            #     name = 'BN-' + name[1:-1]
            #     new_lines.append('name: "%s"' % name)
            #     continue

            line = line.strip()
            if line:
                new_lines.append(line)

#            if line.startswith('layer'):
#                is_layer = True

            if line.startswith('type:') and not layer_type:
                layer_type = line.split(':')[1]
                if not layer_type:
                    layer_type = lines[i + 1].strip()
                layer_type = layer_type.strip()
                layer_type = layer_type[1:-1]

            if line.startswith('name:'):
                layer_name = line.split(':')[1]
                if not layer_name:
                    layer_name = lines[i + 1].strip()
                layer_name = layer_name.strip()
                layer_name = layer_name[1:-1]

            if line.startswith('bottom:'):
                bottom_name = line.split(':')[1].strip()
                bottom_blobs.append(bottom_name[1:-1])

            if line.startswith('top:'):
                top_name = line.split(':')[1].strip()
                top_blobs.append(top_name[1:-1])

            if line.startswith('lr_mult'):
                has_lr_param = True

            if line.startswith('bias_term'):
                if 'true' in line:
                    has_bias_term = True

        print '\n-----------------------'
        print 'layer_name: ', layer_name
        print 'layer_type: ', layer_type
        print 'has_lr_param: ', has_lr_param
        print 'has_bias_term: ', has_bias_term
        print 'bottom_blobs: ', bottom_blobs
        print 'top_blobs: ', top_blobs

        # write the updated prototxt
        for line in new_lines:
            write_line = ''
            if line.endswith('}'):
                tab_cnt -= 1

            if line.startswith('layer'):
                write_line += '\n'

            write_line += '  ' * tab_cnt + line + '\n'

            fp.write(write_line)
            if line.endswith('{'):
                tab_cnt += 1

        if layer_type == 'Convolution':
            bn_layer = append_bn_layer(layer_name, top_blobs[0])
            fp.write(bn_layer)

            # if layer_name not in skip_relu_conv_layers:
            scale_layer = append_scale_layer(
                layer_name + '_bn', top_blobs[0])
            fp.write(scale_layer)

        if layer_type == 'Eltwise':
            relu_layer = append_relu_layer(layer_name, top_blobs[0])
            fp.write(relu_layer)


def main(prototxt_fn, output_fn):
    fp = open(prototxt_fn, 'r')
    fp_out = open(output_fn, 'w')

    section_lines = []

    left_bracket_cnt = 0
    right_bracket_cnt = 0

    for line in fp:
        line = line.strip()
        if not line:
            continue

        # skip line 'param_str' in python layer
        if line.startswith('param_str'):
            section_lines.append(line)
        else:
            if not line.startswith('#'):
                spl, cnt0, cnt1 = split_lines_with_bracket(line)
                left_bracket_cnt += cnt0
                right_bracket_cnt += cnt1
            else:
                spl = [line]

            # skip lines not between two brackets ('{' and '}')
            if left_bracket_cnt < 1:
                # rename the Net
                if line.startswith('name:'):
                    name = line.split(':')[1].strip()
                    name = 'BN-' + name[1:-1]
                    fp_out.write('name: "%s"\n' % name)
                else:
                    fp_out.write(line)
                continue
            else:
                section_lines.extend(spl)

        if left_bracket_cnt > 0 and left_bracket_cnt == right_bracket_cnt:
            write_section(fp_out, section_lines, left_bracket_cnt)
            left_bracket_cnt = 0
            right_bracket_cnt = 0
            section_lines = ['']

    fp.close()
    fp_out.close()


if __name__ == '__main__':
    prototxt_fn = './sphereface_20_model.prototxt'
    if len(sys.argv) > 1:
        prototxt_fn = sys.argv[1]

    output_fn = osp.splitext(prototxt_fn)[0] + '_add_bn.prototxt'
    if len(sys.argv) > 2:
        output_fn = sys.argv[2]

    main(prototxt_fn, output_fn)
