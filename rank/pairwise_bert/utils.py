# -*- coding: utf-8 -*-
# encoding:utf-8
# @File  : utils.py
# @Author: liushuaipeng
# @Date  : 2019/12/20 15:53
# @Desc  : 
import os
import sys
import time

import numpy as np
import tensorflow as tf


if sys.version_info.major == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')


def selectGPU():
    cur = time.time()
    while True:
        os.system(
            'nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp_{}'.format(cur))

        with open('tmp_{}'.format(cur), 'r') as f:
            lines = f.readlines()
            if len(lines) == 0:
                print('没有可用显卡,请确认nvidia-smi -q -d Memory |grep -A4 GPU 命令可用,程序将使用cpu。。。')
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                os.system('rm tmp_{}'.format(cur))
                # exit(0)
                return False

            memory_gpu = [int(x.split()[2]) for x in lines]
            device_no = np.argmax(memory_gpu)
            if int(memory_gpu[device_no]) > 10000:
                print('程序将选用显卡{}，剩余显存{}M'.format(device_no, memory_gpu[device_no]))
                os.environ['CUDA_VISIBLE_DEVICES'] = str(device_no)

                config = tf.ConfigProto(log_device_placement=False)
                config.gpu_options.per_process_gpu_memory_fraction = 1.0
                sess = tf.Session(config=config)
                # K.set_session(sess)
                os.system('rm tmp_{}'.format(cur))
                return True
            else:
                print('最大显存不足，显卡{},显存：{},稍后将重试'.format(device_no, memory_gpu[device_no]))
                time.sleep(30)
                os.system('rm tmp_{}'.format(cur))


