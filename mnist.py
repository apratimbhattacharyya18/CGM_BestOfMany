import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.io as sio
import scipy.misc as smisc
import scipy.signal as sig
import numpy as np
import tensorflow as tf
import math
import sys
import random
import os
import scipy.cluster
import argparse
from tqdm import tqdm

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Lambda, RepeatVector, Dropout, Activation, Flatten
from keras.layers.merge import dot, add, multiply, concatenate
from keras.engine import Layer
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam
from keras import backend as K

from utils import Choose

def get_kl_test_batch():
	idx_list = range(x_data.shape[0]);
	random.shuffle(idx_list);
	x_batch = x_data[idx_list[0:1000],:]
	y_batch = y_data[idx_list[0:1000],:]
	xy_batch = np.concatenate( [x_batch,y_batch], axis = 1 );
	x_batch = np.expand_dims(x_batch,axis=1);
	x_batch = np.repeat(x_batch, 1, axis=1);
	y_batch = np.expand_dims(y_batch,axis=1);
	y_batch = np.repeat(y_batch, 1, axis=1);
	xy_batch = np.expand_dims(xy_batch,axis=1);
	xy_batch = np.repeat(xy_batch, 1, axis=1);
	xy_batch = np.reshape(xy_batch,(xy_batch.shape[0]*1,input_seq + output_seq,2));
	return xy_batch

def get_test_batches():
	x_batch = x_data_test
	y_batch = y_data_test
	tbatch_size = x_data_test.shape[0]
	x_batch = np.expand_dims(x_batch,axis=1);
	x_batch = np.repeat(x_batch, test_samples, axis=1);
	y_batch = np.expand_dims(y_batch,axis=1);
	y_batch = np.repeat(y_batch, test_samples, axis=1);
	x_batch = np.reshape(x_batch,(tbatch_size*test_samples,input_seq,2));
	y_batch = np.reshape(y_batch,(tbatch_size*test_samples,output_seq,2));
	return ( x_batch, y_batch)

def get_batch_gen():
	global x_data, y_data, batch_size, input_seq, output_seq
	while 1:
		for i in xrange((x_data.shape[0]/batch_size)):
			idx = random.randint(0,(x_data.shape[0]/batch_size)-1)
			x_batch = x_data[idx*batch_size:idx*batch_size+batch_size,:]
			y_batch = y_data[idx*batch_size:idx*batch_size+batch_size,:]
			xy_batch = np.concatenate( [x_batch,y_batch], axis = 1 );
			x_batch = np.expand_dims(x_batch,axis=1);
			x_batch = np.repeat(x_batch, train_samples, axis=1);
			y_batch = np.expand_dims(y_batch,axis=1);
			y_batch = np.repeat(y_batch, train_samples, axis=1);
			xy_batch = np.expand_dims(xy_batch,axis=1);
			xy_batch = np.repeat(xy_batch, train_samples, axis=1);
			x_batch = np.reshape(x_batch,(batch_size*train_samples,input_seq,2));
			y_batch = np.reshape(y_batch,(batch_size*train_samples,output_seq,2));
			xy_batch = np.reshape(xy_batch,(batch_size*train_samples,input_seq + output_seq,2));
			yield ({'input_1': x_batch, 'input_2': xy_batch}, {'time_distributed_3': y_batch})


def load_data( train = True ):
	extract_file = './data/MNIST/extracted_data.npy';
	if not os.path.isfile(extract_file):
		data = [];
		max_seq_len = 0;
		for file in tqdm(os.listdir('./data/MNIST/sequences/')):
			if file.endswith('inputdata.txt'):
				curr_seq = np.loadtxt(os.path.join('./data/MNIST/sequences/', file),delimiter=' ')
				data.append(np.cumsum(curr_seq[:,0:2],axis=0));
				if curr_seq.shape[0] > max_seq_len:
					max_seq_len = curr_seq.shape[0]
		for i in xrange(len(data)):
			if data[i].shape[0] < max_seq_len:
				data[i] = np.concatenate([data[i], np.array([data[i][-1,:].tolist(),]*(max_seq_len - data[i].shape[0]))], axis = 0 );
		data = np.array(data);
		np.save(extract_file,data);
	else:
		data = np.load(extract_file);
	if train:
		x = data[0:60000,0:input_seq,:]
		y = data[0:60000,input_seq:,:];
		x = x[0:(x.shape[0]/batch_size)*batch_size,:];
		y = y[0:(x.shape[0]/batch_size)*batch_size,:];

		return ( x, y)
	else:
		x = data[60000:,0:input_seq,:]
		y = data[60000:,input_seq:,:];
		x = x[0:(x.shape[0]/batch_size)*batch_size,:];
		y = y[0:(x.shape[0]/batch_size)*batch_size,:];
		return ( x, y)

z_log_var = 0;
z_mean = 0;
z = 0;

def get_kl_divg( kl_pred ):
	p_z_mean = kl_pred[:,0:latent_dim];
	p_z_log_var = kl_pred[:,latent_dim:];
	kl_loss = - 0.5 * np.sum(1 + p_z_log_var - np.square(p_z_mean) - np.exp(p_z_log_var), axis=-1)
	return np.mean(kl_loss)

def bms_loss( y_true, y_pred ):
	global output_seq
	y_true = K.reshape( y_true, (batch_size,train_samples,output_seq,2) );
	y_pred = K.reshape( y_pred, (batch_size,train_samples,output_seq,2) );
	rdiff = K.mean(K.square(y_pred - y_true),axis=(2,3));
	rdiff_min = K.min( rdiff, axis = 1);
	return K.mean(rdiff_min)

def kl_activity_reg( args ):
	z_mean = args[:,:latent_dim]
	z_log_var = args[:,latent_dim:]
	kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
	return K.mean(kl_loss)

def sample_z( args ):
	z_mean = args[:,:latent_dim]
	z_log_var = args[:,latent_dim:]
	epsilon = K.random_normal(shape=(K.shape(args)[0], latent_dim), mean=0., stddev=1.)
	return z_mean + K.exp(z_log_var / 2) * epsilon

def get_model(input_shape1,input_shape_latent,out_seq):
	input1 = Input(shape=input_shape1)

	input_latent = Input(shape=input_shape_latent)
	input_latent_ = TimeDistributed(Dense(64,activation='relu'))(input_latent)
	h = LSTM(128, implementation = 1)(input_latent_);

	z_mean_var = Dense(latent_dim*2, activity_regularizer=kl_activity_reg)(h)

	z = Lambda(sample_z, output_shape=(latent_dim,))(z_mean_var)
	z = Choose()(z)

	decoder = TimeDistributed(Dense(32, activation='relu'))(input1)
	decoder = LSTM(48, implementation = 1)(decoder);
	decoder = concatenate([decoder,z]);
	decoder = Dense(64, activation='relu')(decoder);
	decoder = RepeatVector(out_seq)(decoder);
	decoder = LSTM(48, implementation = 1, return_sequences=True)(decoder);
	decoder = TimeDistributed(Dense(2))(decoder)

	full_model = Model(inputs= [input1,input_latent], outputs=decoder)
	kl_model = Model(inputs= [input_latent], outputs=[z_mean_var])

	full_model.compile(optimizer = 'adam', loss = bms_loss)

	return ( full_model, kl_model)

def neg_cond_loglikelihood( y_true, y_pred ):
	rdiffs = 0.5*np.sum(np.square(y_pred - y_true),axis=(2,3));
	rexp = (1.0/np.sqrt(2*np.pi)) * np.exp( -rdiffs );
	#rexp = np.maximum(rexp,1e-60);
	rmean = np.mean(rexp,axis=1);
	rmean = np.maximum(rmean,1e-60);
	return np.mean(-np.log(rmean));

def get_cll(preds):
	preds = np.reshape(preds,(x_data_test.shape[0],test_samples,output_seq,2));
	y_batch_test_ = np.reshape(y_batch_test,(x_data_test.shape[0],test_samples,output_seq,2));

	squared_errors = np.mean( np.square(y_batch_test_ - preds), axis = (2,3) );

	cll = 0.0;
	for i in xrange(0,x_data_test.shape[0],32):
		cll += np.mean(neg_cond_loglikelihood( y_batch_test_[i:i+32,], preds[i:i+32,] ));
	cll = cll / ( x_data_test.shape[0] / 32 );	
	return cll

def get_cmap(n, name='hsv'):
	return plt.cm.get_cmap(name, n)

def km_cluster( preds ):
	n_clusters = 4;
	_data_X = np.reshape(preds,(preds.shape[0],-1));
	centroids,_ = scipy.cluster.vq.kmeans(_data_X,n_clusters)
	idx,_ = scipy.cluster.vq.vq(_data_X,centroids)

	clusters = [[] for _ in xrange(n_clusters)];

	for data_idx in xrange(preds.shape[0]):
		clusters[idx[data_idx]].append(preds[data_idx,:]);

	cluster_means = [];
	for c_idx in xrange(n_clusters):
		cluster_means.append( np.mean( np.array(clusters[c_idx]), axis = 0) );

	return clusters	

def plot_results(preds):
	save_dir = './preds_bms/';

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	for i in xrange(200):

		clustered_samples = km_cluster( preds[i,:] );
		
		plt.style.use('dark_background')
		mpl.style.use('seaborn')
		plt.figure(figsize=(8, 8))
		dpi = 80
		axes = plt.gca()
		axes.set_xlim([0,28])
		axes.set_ylim([0,28])
		axes.axis('off')
		plt.plot(x_data_test[i,:,0].tolist(),x_data_test[i,:,1].tolist(),c='w',linewidth=6)
		for j in xrange(len(clustered_samples)):
			samples_in_curr_cluster = np.array(clustered_samples[j]);
			for k in range(samples_in_curr_cluster.shape[0]):
				plt.plot(samples_in_curr_cluster[k,:,0].tolist(),samples_in_curr_cluster[k,:,1],c='C'+str(j),linewidth=6)
		ax=plt.gca()                           
		ax.set_ylim(ax.get_ylim()[::-1])        
		ax.xaxis.tick_top()                     
		plt.savefig(save_dir + str(i) + '.png')
		plt.close()

def train_loop(plot_results_flag):	
	best_cll = 999999.0;
	for epoch in xrange(1,nepochs):
		
		print('Epoch --- ' + str(epoch))
		full_model.fit_generator(my_gen,steps_per_epoch=10,epochs=1,workers=1); #(x_data.shape[0]/batch_size)

		dummy_xy = np.zeros((x_batch_test.shape[0]*test_samples,input_seq + output_seq,2)).astype(np.float32);
		preds = full_model.predict([x_batch_test,dummy_xy], batch_size = batch_size*test_samples, verbose = 1);
		cll = get_cll(preds);
		
		kl_pred = kl_model.predict(kl_test_batch);
		kl_div = get_kl_divg( kl_pred );

		if cll < best_cll:
			best_cll = cll;
			full_model.save('./saved_bms.h5');

		print( 'Current CLL: ' + str(cll) + ', Current KL divg --- ' + str(kl_div) + ', Best CLL: ' + str(best_cll))

		if plot_results_flag:
			plot_results(preds);

		if nepochs > 100:
			next_test_interval = 1;	


batch_size = 32;
latent_dim = 64;
input_seq = 10;

train_samples = 10;
test_samples = 50;

(x_data, y_data) = load_data(True);

(x_data_test, y_data_test) = load_data(False);

output_seq = y_data.shape[1];

( full_model, kl_model) = get_model((input_seq,2),(input_seq + output_seq,2),output_seq);
my_gen = get_batch_gen();

kl_test_batch = get_kl_test_batch();
( x_batch_test, y_batch_test) = get_test_batches();

nepochs = 200;

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--plot', default='True' ,help='Plot Results')
	args = parser.parse_args()
	plot_results_flag = args.plot;

	train_loop(plot_results_flag);
	

