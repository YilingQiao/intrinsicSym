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
flags.DEFINE_integer('batch_size', 4, 'batch size.')
flags.DEFINE_integer('num_point', 4500, '')

# architecture parameters
flags.DEFINE_integer('num_evecs', 12,
					 'number of eigenvectors used for representation. The first 500 are precomputed and stored in input')
flags.DEFINE_integer('dim_input', 4, '')
flags.DEFINE_integer('dim_constraint', 40, '')

# data parameters
flags.DEFINE_string('train_dir', '../data/annotated/train/', '')
flags.DEFINE_string('test_dir', '../data/annotated/test/', '')
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

global train_models_input, train_models_sign, train_models_name, train_num_models, train_order, step
global test_models_input, test_models_sign, test_models_name, test_num_models, test_order
global best_acc, best_ss

		
def load_models_to_ram(mat_dir,is_test=True):
	
	models_sign       = []
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
		#print(os.path.splitext(filename)[0])
		temp_evecs   = sio.loadmat(thisname)['model_evecs'][:,:FLAGS.num_evecs]
		temp_sign    = sio.loadmat(thisname)['model_sign']

		file_pos     = np.zeros((FLAGS.num_point,3))
		file_evecs   = np.zeros((FLAGS.num_point,FLAGS.num_evecs))
		file_sign = temp_sign

		this_point = np.shape(temp_evecs)[0]
		if this_point>FLAGS.num_point:
			file_evecs= temp_evecs[:FLAGS.num_point,:]

		else:
			file_evecs[:this_point,:] = temp_evecs


	
		for iEig in range(min(FLAGS.num_evecs,np.shape(file_evecs)[1])):
			#print(np.shape(np.abs(file_evecs[:,1:4])), np.shape(file_evecs[:,iEig:iEig+1]))
			#print(np.shape(np.concatenate([file_evecs[:,1:3],file_evecs[:,iEig:iEig+1]],axis=1)))
			#print(np.shape(file_evecs))
			if is_test == False:
				if file_sign[0, iEig] == 0:
					continue
			models_input.append(np.concatenate([file_evecs[:,0:3],file_evecs[:,iEig:iEig+1]],axis=1))
			models_sign.append((file_sign[0,iEig]+1)/2) 
		all_names.append(thisname) 
		count = count + 1
	num_models = count
	all_models_sign = np.concatenate([np.expand_dims(x,0) for x in models_sign],axis=0)
	#all_models_sign = np.concatenate([np.resize(x[:,:FLAGS.num_evecs],(FLAGS.num_evecs)) for x in models_sign],axis=0)
	print(np.shape(all_models_sign))
	all_models_input = np.concatenate([np.expand_dims(x,0) for x in models_input],axis=0)
	#print(all_names)
	return all_models_sign, all_models_input, all_names, num_models


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
			train_models_sign, train_models_input, train_models_name, train_num_models = load_models_to_ram(FLAGS.train_dir,is_test=False)
			global best_ss, best_acc
			test_models_sign, test_models_input, test_models_name, test_num_models = load_models_to_ram(FLAGS.test_dir,is_test=True)
			
			best_acc = np.zeros((test_num_models))
			best_ss  = np.zeros((test_num_models, FLAGS.num_evecs))
			
			def train_one_epoch(epoch_num):
				is_training = True
				batch_size =FLAGS.batch_size

				# shuffle the training set
				cur_sign, cur_input, order = shuffle_data(train_models_sign,train_models_input)

				num_data = len(cur_input)
				num_batch = num_data // batch_size
				

				total_loss = 0.0
				total_acc = 0.0

				for j in range(num_batch):
					begidx = j * batch_size
					endidx = (j + 1) * batch_size

					feed_dict = {
						model_sign: cur_sign[begidx: endidx], 
						model_input: cur_input[begidx: endidx], 
						phase: is_training, 
						}

					_, my_loss, step, this_sign = sess.run([train_op, net_loss, global_step, my_sign], feed_dict=feed_dict)

					pre = np.argmax(this_sign,axis=1)
					gt  = cur_sign[begidx: endidx].astype(int)
				
					my_acc = np.mean(pre==gt)

					total_loss += my_loss
					total_acc  += my_acc

				total_loss = total_loss * 1.0 / num_batch
				total_acc  = total_acc * 1.0 / num_batch
				print('train - epoch %d: loss = %.4f, acc = %.4f' % (epoch_num, total_loss, total_acc))


				if epoch_num%20==0:
					saver.save(sess, FLAGS.model_dir + '/model.ckpt', global_step=step)
				

			def eval_one_epoch(epoch_num):
				is_training = False
				batch_size  = FLAGS.batch_size
				cur_sign    = test_models_sign
				cur_input   = test_models_input
				num_data    = len(cur_input)
				

				total_loss = 0.0
				total_acc  = 0.0

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
							model_sign: cur_sign[begidx: endidx], 
							model_input: cur_input[begidx: endidx], 
							phase: is_training, 
							}

						my_loss, this_sign = sess.run([net_loss, my_sign], feed_dict=feed_dict)

						pre = np.argmax(this_sign,axis=1)
						gt  = cur_sign[begidx: endidx].astype(int)

						my_acc = np.mean(pre==gt)

						ss_our.append(pre)
						gt_our.append(gt)

						this_loss += my_loss
						this_acc  += my_acc
					this_acc  = this_acc / num_batch
					this_loss = this_loss / num_batch

					if True:#this_acc > best_acc[model_i]:
						best_acc[model_i] = this_acc
						for j in range(num_batch):
							best_ss[model_i,j*batch_size:(j+1)*batch_size]=ss_our[j]
					total_loss += this_loss 
					total_acc  += this_acc  

				total_loss = total_loss * 1.0 / test_num_models 
				total_acc  = np.mean(best_acc)
				print('test - epoch %d: loss = %.4f, acc = %.4f' % (epoch_num, total_loss, total_acc))

				sio.savemat('S.mat',{'name':test_models_name, 'ss_our':best_ss, 'gt_our':gt_our, 'per_acc':best_acc})

			print('starting training loop...')
			while iteration < FLAGS.max_train_iter:
				if iteration%3==0:
					eval_one_epoch(iteration)
				start_time = time.time()
				train_one_epoch(iteration)
				duration = time.time() - start_time

				print(duration)
				iteration += 1


def main(_):
	run_training()


if __name__ == '__main__':
	tf.app.run()
