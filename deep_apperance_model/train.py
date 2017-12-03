from data_input import DataLoader
import tensorflow as tf
import tensorflow.contrib.slim as slim
import argparse
import os
import time
from appearance_network import AppearanceNetwork

def build_graph(args, global_step, lr, loss):
	tf.summary.scalar('learning_rate', lr)
	
	# Compute gradients.
	opt = tf.train.AdamOptimizer(lr)
	
	grads = tf.gradients(loss, tf.trainable_variables())
	grads, _ = tf.clip_by_global_norm(grads, args.grad_clip)

	# grads = opt.compute_gradients(loss)

	# Apply gradients.
	apply_gradient_op = opt.apply_gradients(zip(grads, tf.trainable_variables()), global_step=global_step)

	# Add histograms for trainable variables.
	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)

	# Add histograms for gradients.
	# for grad, var in grads:
		# if grad is not None:
			# tf.summary.histogram(var.op.name + '/gradients', grad)

	with tf.control_dependencies([apply_gradient_op]):
		train_op = tf.no_op(name='train')

	return train_op

def train(args):
	model = AppearanceNetwork(args)

	save_directory = './save/'
	log_file_path = './training.log'
	log_file = open(log_file_path, 'w')

	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True

	with tf.Graph().as_default():
		global_step = tf.Variable(0, name='global_step', trainable=False)

		image_patches_placeholder = tf.placeholder(tf.float32, shape=[args.batch_size, 7, 128, 64, 3])

		labels_placeholder = tf.placeholder(tf.float32, shape=[args.batch_size])

		lr = tf.Variable(args.base_learning_rate, trainable=False, name="learning_rate")

		features, logits = model.inference(image_patches_placeholder)
		
		loss = model.cross_entropy_loss(logits, labels_placeholder)

		train_op = build_graph(args, global_step, lr, loss)

		sess = tf.Session()

		saver = tf.train.Saver(max_to_keep=100)

		ckpt = tf.train.get_checkpoint_state('./save')
		if ckpt is None:
			init = tf.global_variables_initializer()
			sess.run(init)
			if args.pretrained_ckpt_path is not None:
				# slim.get_or_create_global_step()
				init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
					args.pretrained_ckpt_path, slim.get_variables_to_restore(exclude=
						["lstm", "fc_layer", "loss", "learning_rate", "softmax", "global_step"]))
				sess.run(init_assign_op, feed_dict=init_feed_dict)
		else:
			print 'Loading Model from ' + ckpt.model_checkpoint_path
			saver.restore(sess, ckpt.model_checkpoint_path)

		best_epoch = -1
		best_loss_epoch = 0.0
		for curr_epoch in range(args.num_epoches):
			training_loss_epoch = 0.0
			valid_loss_epoch = 0.0

			############################################# Training process ######################################
			print 'Training epoch ' + str(curr_epoch + 1) + '........................'
			training_data_loader = DataLoader(is_valid=False)

			if curr_epoch % 10 == 0:
				sess.run(tf.assign(lr, args.base_learning_rate * (args.decay_rate ** curr_epoch / 10)))

			training_data_loader.shuffle()
			training_data_loader.reset_pointer()

			for step in range(training_data_loader.num_batches):
				start_time = time.time()

				image_patches, labels = training_data_loader.next_batch()

				_, loss_batch = sess.run([train_op, loss], feed_dict={
						image_patches_placeholder: image_patches,
						labels_placeholder: labels})

				end_time = time.time()
				training_loss_epoch += loss_batch
				print("Training {}/{} (epoch {}), train_loss = {:.8f}, time/batch = {:.3f}"
					.format(
						step + 1,
						training_data_loader.num_batches,
						curr_epoch + 1,
						loss_batch, end_time - start_time))

			print 'Epoch ' + str(curr_epoch + 1) + ' training is done! Saving model...'
			checkpoint_path = os.path.join(save_directory, 'model.ckpt')
			saver.save(sess, checkpoint_path, global_step=global_step)

			############################################# Validating process ######################################
			print 'Validating epoch ' + str(curr_epoch + 1) + '...........................'
			valid_data_loader = DataLoader(is_valid=True)

			valid_data_loader.shuffle()
			valid_data_loader.reset_pointer()
			for step in range(valid_data_loader.num_batches):
				start_time = time.time()

				image_patches, labels = valid_data_loader.next_batch()

				loss_batch = sess.run(loss, feed_dict={
						image_patches_placeholder: image_patches,
						labels_placeholder: labels}
						)

				end_time = time.time()
				valid_loss_epoch += loss_batch
				print("Validating {}/{} (epoch {}), valid_loss = {:.8f}, time/batch = {:.3f}"
					.format(
						step + 1,
						valid_data_loader.num_batches,
						curr_epoch + 1, 
						loss_batch, end_time - start_time))

			# Update best valid epoch
			if best_epoch == -1 or best_loss_epoch > valid_loss_epoch:
				best_epoch = curr_epoch + 1
				best_loss_epoch = valid_loss_epoch

			log_file.write('epoch ' + str(curr_epoch + 1) + '\n')
			log_file.write(str(curr_epoch + 1) + ',' + str(training_loss_epoch) + '\n')
			log_file.write(str(curr_epoch + 1) + ',' + str(valid_loss_epoch) + '\n')
			log_file.write(str(best_epoch) + ',' + str(best_loss_epoch) + '\n')

		log_file.close()



def main():
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--is_training', type=bool, default=True,
						help='is this training process?')

	parser.add_argument('--num_epoches', type=int, default=100,
						help='the number of epoches to train')
	
	parser.add_argument('--batch_size', type=int, default=64,
						help='batch size for training')
	
	parser.add_argument('--base_learning_rate', type=float, default=0.02,
						help='learning_rate')
	
	parser.add_argument('--decay_rate', type=float, default=0.1,
						help='decay rate for learning rate')

	parser.add_argument('--lstm_num_units', type=int, default=128,
						help='the number of units in a lstm cell')
	
	parser.add_argument('--lstm_num_cells', type=int, default=6,
						help='the number of cells in lstm sequence')
	
	parser.add_argument('--l2_normalize', type=bool, default=True,
						help='if l2 normalize output features')

	parser.add_argument('--pretrained_ckpt-path', type=str, default='./resources/networks/mars-small128.ckpt')

	parser.add_argument('--grad_clip', type=float, default=1.,
						help='the grad clip')
	
	args = parser.parse_args()
	
	train(args)

if __name__ == '__main__':
    main()