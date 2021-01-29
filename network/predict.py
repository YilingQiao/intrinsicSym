import os
import time
import tensorflow as tf
import numpy as np
import scipy.io as sio
import pdb

from models import SignNet


flags = tf.app.flags
FLAGS = flags.FLAGS

# training params
flags.DEFINE_float('learning_rate', 1e-4, 'initial learning rate.')
flags.DEFINE_integer('batch_size', 1, 'batch size.')
flags.DEFINE_integer('num_point', 4500, '')

# architecture parameters
flags.DEFINE_integer('num_evecs', 17,
					 'number of eigenvectors used for representation. The first 500 are precomputed and stored in input')
flags.DEFINE_integer('dim_input', 4, '')
flags.DEFINE_integer('dim_constraint', 40, '')

# data parameters
flags.DEFINE_string('pred_dir', '../data/demo/', '')
flags.DEFINE_string('model_dir', './Results/train_inter_k_flag', 'directory to save models and results')
flags.DEFINE_integer('max_train_iter', 150, '')
flags.DEFINE_integer('save_summaries_secs', 60, '')
flags.DEFINE_integer('save_model_secs', 1200, '')
flags.DEFINE_string('master', '', '')

test_ratio = 7
high_eval_acc = 0

def shuffle_data(sign, input):
    idx = np.arange(len(input))
    np.random.shuffle(idx)
    return sign[idx], input[idx], idx

global test_models_input, test_models_name, test_num_models, test_order
global best_ss

		
def load_models_to_ram(mat_dir,is_test=True):
	
	models_input      = []
	all_names         = []
	f_list = os.listdir(mat_dir)
	count = 0
	idx = -1
	for filename in f_list:
		idx = idx+1
		print(count)
		if count > 20000: #control the num of data used
			break

		thisname    = mat_dir + os.path.splitext(filename)[0]  + '.mat'
		print(os.path.splitext(filename)[0][0:3])

		if os.path.splitext(filename)[1] != '.mat':
			continue
		print(thisname)
		temp_evecs   = sio.loadmat(thisname)['model_evecs'][:,:FLAGS.num_evecs]

		file_evecs   = np.zeros((FLAGS.num_point,FLAGS.num_evecs))

		this_point = np.shape(temp_evecs)[0]
		if this_point>FLAGS.num_point:
			file_evecs= temp_evecs[:FLAGS.num_point,:]

		else:
			file_evecs[:this_point,:] = temp_evecs


	
		for iEig in range(min(FLAGS.num_evecs,np.shape(file_evecs)[1])):
			models_input.append(np.concatenate([file_evecs[:,0:3],file_evecs[:,iEig:iEig+1]],axis=1))
		
		all_names.append(thisname) 
		count = count + 1
	num_models = count
	all_models_input = np.concatenate([np.expand_dims(x,0) for x in models_input],axis=0)
	#print(all_names)
	return all_models_input, all_names, num_models


def run_training():

	print('model_dir=%s' % FLAGS.model_dir)
	if not os.path.isdir(FLAGS.model_dir):
		os.makedirs(FLAGS.model_dir) 
	print('num_evecs=%d' % FLAGS.num_evecs)

	print('building graph...')
	with tf.Graph().as_default():
		NUM_POINT = FLAGS.num_point

		# inpute vector (x,y,z,eigen_function_value)
		model_input = tf.placeholder(tf.float32, shape=(None, FLAGS.num_point, FLAGS.dim_input), name='model_input')
		# sign of thie function
		model_sign  = tf.placeholder(tf.int32, shape=(None), name='model_sign')
		# train or test
		phase = tf.placeholder(dtype=tf.bool, name='phase')

		# network module
		net_loss, my_sign, merged = SignNet(phase, model_input, model_sign)
		
		summary = tf.summary.scalar("num_evecs", float(FLAGS.num_evecs))

		global_step = tf.Variable(0, name='global_step', trainable=False)

		optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

		train_op = optimizer.minimize(net_loss, global_step=global_step)

		saver = tf.train.Saver(max_to_keep=10)

		sv = tf.train.Supervisor(logdir=FLAGS.model_dir,
								 init_op=tf.global_variables_initializer(),
								 local_init_op=tf.local_variables_initializer(),
								 global_step=global_step,
								 save_summaries_secs=FLAGS.save_summaries_secs,
								 save_model_secs=FLAGS.save_model_secs,
								 summary_op=None,
								 saver=saver)

		writer = sv.summary_writer


		print('starting session...')
		iteration = 0
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True

		with sv.managed_session(config=config,master=FLAGS.master) as sess:

			print('loading data to ram...')
			global best_ss
			test_models_input, test_models_name, test_num_models = load_models_to_ram(FLAGS.pred_dir,is_test=True)
			
			best_ss  = np.zeros((test_num_models, FLAGS.num_evecs))
			

			def predict_one_epoch():
				is_training = False
				batch_size  = FLAGS.batch_size
				cur_input   = test_models_input
				num_data    = len(cur_input)
				


				gt_our = []

				for model_i in range(test_num_models):
					num_batch = FLAGS.num_evecs // batch_size
					ss_our = []

					this_loss = 0
					this_acc  = 0

					for j in range(num_batch):
						begidx = j * batch_size + model_i * FLAGS.num_evecs
						endidx = (j + 1) * batch_size + model_i * FLAGS.num_evecs

						feed_dict = {
							model_input: cur_input[begidx: endidx], 
							phase: is_training, 
							}

						this_sign = sess.run(my_sign, feed_dict=feed_dict)

						pre = np.argmax(this_sign,axis=1)

						ss_our.append(pre)


					if True:
						for j in range(num_batch):
							best_ss[model_i,j*batch_size:(j+1)*batch_size]=ss_our[j]

				sio.savemat('../data/predict/S.mat',{'name':test_models_name, 'ss_our':best_ss})

			predict_one_epoch()


def main(_):
	run_training()


if __name__ == '__main__':
	tf.app.run()
