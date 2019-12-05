import cv2
import os
import argparse
import numpy as np
from monodepth_model import *
import pdb
import time
from utils.io import save_pfm


monodepth_parameters = namedtuple('parameters',
                        'config_path, '
                        'dropout, '
                          )

parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')
parser.add_argument('--input_image',           type=str,   help='path to the input', default='')
parser.add_argument('--checkpoint_path',       type=str,   help='path to a specific checkpoint to load', default='')
parser.add_argument('--dropout',                           help='if use dropout', action='store_true')
parser.add_argument('--config_path',           type=str,   help='configuration file path', default='model.config')
parser.add_argument('--max_depth',             type=float, help='maximum depth for visualization (default 100m)', default=100)
args = parser.parse_args()

params = monodepth_parameters(
    dropout=args.dropout,
    config_path=args.config_path,)

# input
#left_numpy = cv2.imread('/data/gengshay/KITTI/2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000069.jpg')
#left_numpy = cv2.imread('/data/gengshay/nyuv2/JPEGImages/0001.jpg')
left_numpy = cv2.imread(args.input_image)
origin_shape = left_numpy.shape[:2]

#shape = [384,1152]
#shape = [480,640]
h = left_numpy.shape[0] // 32 * 32
w = left_numpy.shape[1] // 32 * 32
if h < left_numpy.shape[0]: h += 32
if w < left_numpy.shape[1]: w += 32
shape = [h,w]

left_numpy = cv2.resize(left_numpy, (shape[1],shape[0]))
left_numpy = left_numpy[np.newaxis,:,:,::-1].astype(float)/255


# model
left = tf.placeholder(tf.float32, (1,None,None,3))
model = MonodepthModel(params, 'test', left, shape)
outlist = [model.most_likely,model.expectation,model.entropy, model.resp]

# SESSION
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

# SAVER
train_saver = tf.train.Saver()

# INIT
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
coordinator = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

# RESTORE
restore_path = args.checkpoint_path.split(".")[0]
train_saver.restore(sess, restore_path)


# RUN
output= sess.run( outlist , feed_dict={left: left_numpy})
depth_ml = output[0].squeeze()
depth_exp = output[1].squeeze()
entropy = output[2].squeeze()
entropy[np.isnan(entropy)] = 0
resp = output[3].squeeze()

# output
depth_ml = cv2.resize(depth_ml, origin_shape[::-1])
depth_exp = cv2.resize(depth_exp, origin_shape[::-1])
entropy = cv2.resize(entropy, origin_shape[::-1])
cv2.imwrite('./output-ml.png',   depth_ml/args.max_depth * 255.)
cv2.imwrite('./output-exp.png', depth_exp/args.max_depth * 255.)
cv2.imwrite('./output-entropy.png',entropy/entropy.max()*255)

# store unscaled depth image in .pfm format
# .pfm can be visualized using cvkit: http://vision.middlebury.edu/stereo/code/
with open('./disp0.pfm','w') as f:
    save_pfm(f,np.clip(depth_exp,1,args.max_depth)[::-1,:])
