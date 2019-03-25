# from __future__ import division
from ipywidgets import interact
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import shutil
from utils import *
from HOG import *
import argparse
import IPython

def identityFeatureExtractor(x):
	return x

def customFeatureExtractor(x):
	'''
	OPTIONAL:
	Write your own custom feature extractor.
	It takes in x, a N x 2 array and outputs features of your design.
	It should output a N x M array where M is the number of features
	For example, it can output x1, x2, x1*x2

	add the --feature custom flag when you run this script!
	'''
	### Your code starts here ###
	phi_x = np.hstack([x, x**2, x[:,:1]*x[:,1:2]])
	#############################
	return phi_x

def svm(data, load=False, feature_extractor=identityFeatureExtractor):
	# unpack data
	x_train = data['x_train']
	y_train = data['y_train']
	x_eval = data['x_eval']
	y_eval = data['y_eval']
	x_pred = data['x_pred']
	y_true = data['y_true'] 

	model_dir = "training_checkpoints"

	# delete files if you are training
	if not load: shutil.rmtree(model_dir + "/" + data['name'], ignore_errors=True)

	# hyperparameters
	hps = tf.contrib.training.HParams(
		train_batch_size = 32,
		eval_batch_size = 32,
		######### Your code starts here #########
		# Select your learning rate and lambda value here 
		lr = 0.01,# 0.08 solved
		lam = 0.001	# 0.15 solved
		######### Your code ends here #########
		)

	# this function details your core learning model 
	def model_fn(features, labels, mode, params):
		predictions = None
		loss = None
		train_op = None
		eval_metric_ops = None 
		export_outputs = None
		x = tf.to_float(features["x"])
		if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
			y = tf.to_float(labels["y"])
			with tf.variable_scope("svm"):
				W = tf.get_variable("weights", dtype=tf.float32, shape=(x.shape[-1],1))
				b = tf.get_variable("bias", dtype=tf.float32, shape=(1,))

				######### Your code starts here #########
				'''
				Compute y_est, and the loss function.
				Beware of variable types, (is it an int, or a float?).
				i.e., y_est = ...., loss = .....
				'''
				y_est = tf.matmul(x, W) - b  # As mentioned in the Problem Statement Description
				loss = tf.reduce_mean(tf.maximum(0.0, 1.0-y*y_est)) + params.lam * tf.norm(W)**2 
				######### Your code ends here #########

				accuracy = tf.reduce_mean(tf.to_float(y*y_est > 0))
				eval_metric_ops = {"accuracy": tf.metrics.mean(accuracy)}
				if mode == tf.estimator.ModeKeys.TRAIN:
					opt = tf.train.GradientDescentOptimizer(learning_rate=params.lr)
					train_op = opt.minimize(loss, tf.train.get_global_step())
				return tf.estimator.EstimatorSpec(mode, 
										  predictions=predictions, 
										  loss=loss,
										  train_op=train_op,
										  eval_metric_ops=eval_metric_ops)

		elif mode == tf.estimator.ModeKeys.PREDICT:
			predictions = {}
			with tf.variable_scope("svm"):
				W = tf.get_variable("weights", shape=(x.shape[-1],1), dtype=tf.float32)
				b = tf.get_variable("bias", shape=(1,), dtype=tf.float32)

				######### Your code starts here #########
				'''
				Compute the y_est (estimate of y), the label.
				i.e., y_est = ...., label = .....
				'''
				y_est = tf.matmul(x,W) - b # Same as last time
				label = tf.sign(y_est)
				######### Your code ends here #########

				predictions["y_est"] = y_est
				predictions["labels"] = label
				export_outputs = {'prediction': tf.estimator.export.PredictOutput(predictions)}
				return tf.estimator.EstimatorSpec(mode, 
										  predictions=predictions, 
										  export_outputs=export_outputs)

		


	def predict_input_function(x_dict):
		return tf.estimator.inputs.numpy_input_fn(x=x_dict,
												  num_epochs=1,
												  shuffle=False)

	rc = tf.estimator.RunConfig().replace(model_dir=model_dir + "/" + data['name'],
										  tf_random_seed=None,
										  save_summary_steps=100,
										  save_checkpoints_steps=1000,
										  keep_checkpoint_max=1000,
										  log_step_count_steps=1000)


	# define your estimator model, using model_fn, hps, rc, and model_dir
	estimator_model = tf.estimator.Estimator(model_fn, 
											 params=hps, 
											 config=rc,
											 model_dir=model_dir + "/" + data['name'],
											 )
	# if you want to load a model, you just need to return the estimator model
	# there is no need to do the data input stuff.
	if load:
		return estimator_model

	# function to set up your training data
	# x_train, y_train must by numpy arrays
	train_input_function = tf.estimator.inputs.numpy_input_fn(x={"x":feature_extractor(x_train)},
															  y={"y":y_train},
															  batch_size=hps.train_batch_size,
															  num_epochs=None,
															  shuffle=True)
	# function to set up your eval data
	eval_input_function = tf.estimator.inputs.numpy_input_fn(x={"x":feature_extractor(x_eval)},
															 y={"y":y_eval},
															 batch_size=hps.eval_batch_size,
															 num_epochs=1,
															 shuffle=False)


	# preparing your data for the estimator model
	train_spec = tf.estimator.TrainSpec(input_fn=train_input_function, max_steps=10000)
	eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_function, steps=100, start_delay_secs=0)
	# remove any logs 
	
	# CARRY OUT THE TRAINING!!
	tf.estimator.train_and_evaluate(estimator_model, train_spec, eval_spec)

	print("Traing and evaluate completed")
	# save the model
	if data['name'] == 'hog': 
		with tf.Graph().as_default():
			serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
				{"x": tf.placeholder(tf.float32, shape=[None, 1152], name="x"),
				 "y": tf.placeholder(tf.float32, shape=[None, 1], name="y")})
			estimator_model.export_savedmodel("training_checkpoints/" + data['name'] + "/model", serving_input_receiver_fn,
							)
	# return the predicted labels.
	return [y["labels"][0] for y in estimator_model.predict(predict_input_function({"x":feature_extractor(x_pred)}), predict_keys = "labels")] 



def get_hog_data():
	'''
	This function calls the "hog_descriptor" function to obtain hog features for a dataset.
	Luckily, we have implemented hog_descriptor for you. 
	"hog_descriptor" uses tensorflow to compute the features, therefore, it outputs the hog feature 
	tensor after evaluating the computation graph with the given input.
	Recall that in order to access values at a particular node of a graph, you must:
		(1) first define a session
		(2) use .eval() on the tensor of interest to obtain the values as a numpy array.
	'''
	with tf.Session() as sess:
		pedestrian_data = np.load("pedestrian_dataset.npz")
		######### Your code starts here #########
		# These should be all be numpy arrays
		train_pos, train_neg = pedestrian_data['train_pos'], pedestrian_data['train_neg'] 
		eval_pos, eval_neg = pedestrian_data['eval_pos'], pedestrian_data['eval_neg']
		test_pos, test_neg = pedestrian_data['test_pos'], pedestrian_data['test_neg']

		x_train = tf_histogram_of_oriented_gradients(np.vstack([train_pos, train_neg]))[1].eval()
		x_eval = tf_histogram_of_oriented_gradients(np.vstack([eval_pos, eval_neg]))[1].eval()
		x_pred = tf_histogram_of_oriented_gradients(np.vstack([test_pos, test_neg]))[1].eval()
		
		# Extract sizes
		train_size, eval_size, test_size = x_train.shape[0], x_eval.shape[0], x_pred.shape[0]


		y_train = np.array([1]*train_pos.shape[0] + [-1]*train_neg.shape[0], dtype=np.float32).reshape(train_size, 1)
		y_eval = np.array([1]*eval_pos.shape[0] + [-1]*eval_neg.shape[0], dtype=np.float32).reshape(eval_size, 1)
		y_true = np.array([1]*test_pos.shape[0] + [-1]*test_neg.shape[0], dtype=np.float32).reshape(test_size, 1)
		######### Your code ends here #########
	return (x_train, y_train), (x_eval, y_eval), (x_pred, y_true)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--type',   type=str, help="toy or hog")
	parser.add_argument('--feature',   type=str, default="identity", help="identity or custom")

	args = parser.parse_args()

	if args.type == "toy":
		x_train, y_train = generate_data(N=5000)
		x_eval, y_eval = generate_data(N=1000)
		x_pred, y_true = generate_data(N=1000)
		data = {}
		data['name'] = 'toy'
		data['x_train'] = x_train
		data['y_train'] = y_train
		data['x_eval'] = x_eval
		data['y_eval'] = y_eval
		data['x_pred'] = x_pred
		data['y_true']  = y_true

		feature_extractor = customFeatureExtractor if args.feature == "custom" else identityFeatureExtractor
		predicted_labels = svm(data, feature_extractor=feature_extractor)

		maybe_makedirs("../plots")
		plt.figure(figsize=(10, 3))
		plt.subplot(1,2,1)
		c = ['powderblue' if lb == 1 else 'indianred' for lb in y_train]
		plt.scatter(x_train[:,0], x_train[:,1], c = c, alpha=0.5, s=50)
		plt.title("Dataset to be classified using %s features" % args.feature)

		plt.subplot(1,2,2)
		c = ['powderblue' if lb == 1 else 'indianred' for lb in predicted_labels]
		plt.scatter(x_pred[:,0], x_pred[:,1], c=c, s=50, alpha=0.5)
		# misclassified data
		d = predicted_labels - y_true[:,0]
		misclass_idx = np.where(d!= 0)[0]
		c = ['red' if lb == 2 else 'blue' for lb in d[misclass_idx]]
		plt.scatter(x_pred[misclass_idx,0], x_pred[misclass_idx,1], c=c, s=50, alpha=0.8)
		accuracy = 100*(1-len(misclass_idx)/float(x_pred.shape[0]))
		plt.title("Classification results: %.2f%%" % accuracy )
		plt.savefig('../plots/P1_svm.png')
		plt.show()


	elif args.type == "hog":
		(x_train, y_train), (x_eval, y_eval), (x_pred, y_true) = get_hog_data()
		data = {}
		data['name'] = 'hog'
		data['x_train'] = x_train
		data['y_train'] = y_train
		data['x_eval'] = x_eval
		data['y_eval'] = y_eval
		data['x_pred'] = x_pred
		data['y_true']  = y_true
		predicted_labels = svm(data)
		d = predicted_labels - y_true[:,0]
		misclass_idx = np.where(d!= 0)[0]
		np.save("hog_misclass_idx.npy", misclass_idx)
		accuracy = 100*(1 - len(misclass_idx)/float(x_pred.shape[0]))
		print("\n\n\n\nClassification results: %.2f%%\n\n\n" % accuracy)

		maybe_makedirs("../plots")
		plt.figure(figsize=(10, 3))
		plt.subplot(1,2,1)
		c = ['powderblue' if lb == 1 else 'indianred' for lb in y_train]
		plt.scatter(x_train[:,0], x_train[:,1], c = c, alpha=0.5, s=50)
		plt.title("Dataset to be classified using %s features" % args.feature)
		# I added
		plt.subplot(1,2,2)
		c = ['powderblue' if lb == 1 else 'indianred' for lb in predicted_labels]
		plt.scatter(x_pred[:,0], x_pred[:,1], c=c, s=50, alpha=0.5)
		# misclassified data
		d = predicted_labels - y_true[:,0]
		misclass_idx = np.where(d!= 0)[0]
		c = ['red' if lb == 2 else 'blue' for lb in d[misclass_idx]]
		plt.scatter(x_pred[misclass_idx,0], x_pred[misclass_idx,1], c=c, s=50, alpha=0.8)
		accuracy = 100*(1-len(misclass_idx)/float(x_pred.shape[0]))
		plt.title("Classification results: %.2f%%" % accuracy )
		plt.savefig('../plots/P1_svm.png')
		plt.show()
