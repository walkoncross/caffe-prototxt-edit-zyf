# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 02:43:55 2017

@author: zhaoy
"""

import os.path as osp

# prototxt_fn = './train_ssh_resnet50_nolr.prototxt'
# output_fn = osp.splitext(prototxt_fn)[0] + '_add_lr.prototxt'
# prototxt_fn = './train_ssh_resnet101_nolr.prototxt'
# output_fn = osp.splitext(prototxt_fn)[0] + '_add_lr4.prototxt'
# prototxt_fn = './train_fpn_resnet50_nolr.prototxt'
# output_fn = osp.splitext(prototxt_fn)[0] + '_add_lr.prototxt'
prototxt_fn = './train_fpn_resnet101_nolr.prototxt'
output_fn = osp.splitext(prototxt_fn)[0] + '_add_lr.prototxt'

ADD_LR_PARAM = True


def split_lines_with_bracket(line):
    spl = []
    cnt0 = 0
    cnt1 = 0

    pos0 = 0

    line = line.strip()

    for i,ch in enumerate(line):
        if ch=='{':
            spl.append(line[pos0:i+1])
#            spl.append('{')
            pos0 = i+1
            cnt0 += 1
        elif ch=='}':
            spl.append(line[pos0:i])
            spl.append('}')
            pos0 = i+1
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

        # skip the last '}' in the layer
        for i in range(len(lines)-1):
            line = lines[i].strip()
            if line:
                new_lines.append(line)

#            if line.startswith('layer'):
#                is_layer = True

            if line.startswith('type:') and not layer_type:
                layer_type = line.split(':')[1]
                if not layer_type:
                    layer_type = lines[i+1].strip()
                layer_type = layer_type.strip()
                layer_type = layer_type[1:-1]

            if line.startswith('name:'):
                layer_name = line.split(':')[1]
                if not layer_name:
                    layer_name = lines[i+1].strip()
                layer_name = layer_name.strip()
                layer_name = layer_name[1:-1]

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

        ## add lr_param
        if ADD_LR_PARAM and not has_lr_param:
            if (layer_name.startswith('conv1') or
                layer_name.startswith('scale_conv1') or
                layer_name.startswith('res1') or
                layer_name.startswith('res2') or
                layer_name.startswith('bn1') or
                layer_name.startswith('bn2') or
                layer_name.startswith('scale1') or
                layer_name.startswith('scale2')
                    ):
                print '===>', layer_type
                if layer_type=='Convolution' or layer_type=='Scale':
                    #fix lr_mult for weight
                    add_lr_param(new_lines, lr_param_fixed)
                    print 'add lr_param_fixed for weight'
                    #fix lr_mult for bias_term
                    if has_bias_term:
                        add_lr_param(new_lines, lr_param_fixed)
                        print 'add lr_param_fixed for bias'
                elif layer_type=='BatchNorm':
                    for k in range(3):
                        add_lr_param(new_lines, lr_param_fixed_nodecay)
                        print 'add lr_param_fixed_nodecay for BatchNorm'

            elif (layer_name.startswith('res') or
                  layer_name.startswith('bn') or
                  layer_name.startswith('scale')
                   ):
                print '===>', layer_type
                if layer_type=='Convolution' or layer_type=='Scale':
                    add_lr_param(new_lines, lr_param_for_weight_nodecay)
                    print 'add lr_param_for_weight for weight'
                    if has_bias_term:
                        add_lr_param(new_lines, lr_param_for_bias_nodecay)
                        print 'add lr_param_for_bias for bias'

                if layer_type=='BatchNorm':
                    for k in range(3):
                        add_lr_param(new_lines, lr_param_fixed_nodecay)
                        print 'add lr_param_fixed_nodecay for BatchNorm'

        # add the last '}' in the layer
        new_lines.append('}')

        # write the updated prototxt
        for line in new_lines:
            write_line = ''
            if line.endswith('}'):
                tab_cnt -= 1

            if line.startswith('layer'):
                write_line += '\n'

            write_line += '\t'*tab_cnt + line + '\n'

            fp.write(write_line)
            if line.endswith('{'):
                tab_cnt += 1

def main():
    fp = open(prototxt_fn, 'r')
    fp_out = open(output_fn, 'w')

    section_lines=[]

    left_bracket_cnt = 0
    right_bracket_cnt = 0

    for line in fp:
        line = line.strip()
        if not line:
            continue

        if line.startswith('param_str'):
            section_lines.append(line)
        else:
            spl, cnt0, cnt1 = split_lines_with_bracket(line)
            section_lines.extend(spl)
            left_bracket_cnt += cnt0
            right_bracket_cnt += cnt1

        if left_bracket_cnt>0 and left_bracket_cnt==right_bracket_cnt:
            write_section(fp_out, section_lines, left_bracket_cnt)
            left_bracket_cnt = 0
            right_bracket_cnt = 0
            section_lines = ['']

    fp.close()
    fp_out.close()

if __name__=='__main__':
    main()