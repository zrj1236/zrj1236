import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import gzip
import sys
from random import randint
import numpy as np
from numpy import array
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import DenseLayer, DropoutLayer, ReshapeLayer, InputLayer, FlattenLayer, Upscale2DLayer, LocalResponseNormalization2DLayer
floatX = theano.config.floatX
from lasagne.layers import Deconv2DLayer
from lasagne.layers import batch_norm
from lasagne.objectives import categorical_crossentropy, binary_crossentropy, categorical_accuracy, binary_accuracy
import pickle
from PIL import Image
from sklearn import metrics
from sklearn.metrics import roc_auc_score

def AUC(res,label):
	p=[]
	for tp in res[0]:
		tp=list(tp)
		p.append(tp[1])
	fpr, tpr, _ = metrics.roc_curve(label,p,pos_label=1)	
	auc=metrics.auc(fpr, tpr)	
	return auc
	
def offseq_hot_sc_trans(seq1,seq2):
	tpl=[]
	for i in range(0,len(seq1)):
		tp=[0,0,0,0]
		if seq1[i] == 'A':
			tp[0] = 1
		elif seq1[i] == 'T':
			tp[1] = 1		
		elif seq1[i] == 'C':
			tp[2] = 1				
		elif seq1[i] == 'G':
			tp[3] = 1
		if seq2[i] == 'A':
			tp[0] = 1
		elif seq2[i] == 'T':
			tp[1] = 1		
		elif seq2[i] == 'C':
			tp[2] = 1				
		elif seq2[i] == 'G':
			tp[3] = 1			
		tpl.extend(tp)		
	tpl=np.array(tpl)
	tpl=tpl.reshape(1,len(seq1),4)
	return tpl	

def load_off_dataset(filename):
	f1=open(filename,'r')
	m1=f1.readlines()
	Xpos=[]
	Xnet=[]
	for i in range(1,len(m1)):
		p1=m1[i].strip().split('\t')
		if p1[2] == '1':
			Xpos.append(offseq_hot_sc_trans(p1[0],p1[1]))
		if p1[2] == '0':
			Xnet.append(offseq_hot_sc_trans(p1[0],p1[1]))
	Xpos = np.array(Xpos, dtype='int32')	
	Xnet = np.array(Xnet, dtype='int32')	
	return Xpos,Xnet

def load_on_dataset(filename):
	f1=open(filename,'r')
	m1=f1.readlines()
	Xpos=[]
	Xnet=[]
	for i in range(1,len(m1)):
		p1=m1[i].strip().split('\t')
		if p1[1] == '1':
			Xpos.append(offseq_hot_sc_trans(p1[0],p1[0]))
		if p1[1] == '0':
			Xnet.append(offseq_hot_sc_trans(p1[0],p1[0]))
	Xpos = np.array(Xpos, dtype='int32')	
	Xnet = np.array(Xnet, dtype='int32')	
	return Xpos,Xnet

def pos_net_pair(pos,net,size):
	tpind = np.arange(len(pos))
	np.random.shuffle(tpind)
	tpXpos=pos[tpind[:size]]
	tpypos=np.ones(size)
	tpind = np.arange(len(net))
	np.random.shuffle(tpind)
	tpXnet=net[tpind[:size]]
	tpynet=np.zeros(size)
	tptrainX=[]
	tptrainX.extend(tpXpos)		
	tptrainX.extend(tpXnet)		
	tptrainX = np.array(tptrainX, dtype='int32')	
	tptrainy=[]
	tptrainy.extend(tpypos)		
	tptrainy.extend(tpynet)	
	tptrainy = np.array(tptrainy, dtype='int32')
	tpind = np.arange(len(tptrainX))
	np.random.shuffle(tpind)		
	tpX=tptrainX[tpind]
	tpy=tptrainy[tpind]	
	return tpX,tpy

def pos_net_all(pos,net):
	tpXpos=pos[:]
	tpypos=np.ones(len(tpXpos))
	tpXnet=net[:]
	tpynet=np.zeros(len(tpXnet))
	tptrainX=[]
	tptrainX.extend(tpXpos)		
	tptrainX.extend(tpXnet)		
	tptrainX = np.array(tptrainX, dtype='int32')
	tptrainy=[]
	tptrainy.extend(tpypos)		
	tptrainy.extend(tpynet)	
	tptrainy = np.array(tptrainy, dtype='int32')
	tpind = np.arange(len(tptrainX))
	np.random.shuffle(tpind)		
	tpX=tptrainX[tpind]
	tpy=tptrainy[tpind]	
	return tpX,tpy

def Confusion_Matrix(label,result):
	label=np.array(label)
	result=np.array(result)
	TP=float(list(label+result).count(2))
	FN=float(list(label-result).count(1))
	FP=float(list(label-result).count(-1))
	TN=float(list(label+result).count(0))
	acc=(TP+TN)/(TP+FN+FP+TN)
	pre=(TP)/(TP+FP)
	tpr=(TP)/(TP+FN)
	tnr=(TN)/(TN+FP)
	F1=(2*pre*tpr)/(pre+tpr)
	return acc,pre,tpr,tnr,F1

def softmax_transf(res):
	p=[]
	for tp in res[0]:
		tp=list(tp)
		pp=tp.index(max(tp))	
		p.append(pp)
	return p

def build_sc_cnn(input_var=None):
	network = lasagne.layers.InputLayer(shape=(None, 1, 23, 4),input_var=input_var)
	network1 = lasagne.layers.Conv2DLayer(network, num_filters=10, filter_size=(1, 4),nonlinearity=lasagne.nonlinearities.rectify)
	network1 = lasagne.layers.PadLayer(network1, width=[[0,1],[0,0]])
	network2 = lasagne.layers.Conv2DLayer(network, num_filters=10, filter_size=(2, 4),nonlinearity=lasagne.nonlinearities.rectify)
	network2 = lasagne.layers.PadLayer(network2, width=[[0,2],[0,0]])
	network3 = lasagne.layers.Conv2DLayer(network, num_filters=10, filter_size=(3, 4),nonlinearity=lasagne.nonlinearities.rectify)
	network3 = lasagne.layers.PadLayer(network3, width=[[0,3],[0,0]])
	network4 = lasagne.layers.Conv2DLayer(network, num_filters=10, filter_size=(4, 4),nonlinearity=lasagne.nonlinearities.rectify)
	network4 = lasagne.layers.PadLayer(network4, width=[[0,4],[0,0]])
	network = lasagne.layers.ConcatLayer([network1,network2,network3,network4])
	network = lasagne.layers.BatchNormLayer(network)
	network = FlattenLayer(network)
	network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.15),num_units=200,nonlinearity=lasagne.nonlinearities.rectify)
	network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.15),num_units=23,nonlinearity=lasagne.nonlinearities.rectify)
	network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.15),num_units=2,nonlinearity=lasagne.nonlinearities.softmax)	
	return network



filename1='/home/rongyu/project/project/GAN/2019.10.31/data/off_target.CNN.txt'
#filename2='/home/rongyu/project/project/GAN/2019.10.31/data/on_target_293.txt'

Xpos_1,Xnet_1=load_off_dataset(filename1)
#Xpos_2,Xnet_2=load_on_dataset(filename2)

####### off-data ##########
indices=np.arange(len(Xpos_1))
np.random.shuffle(indices)
Xpos_train_1 = Xpos_1[len(indices)/5:] 
Xpos_val_1 = Xpos_1[:len(indices)/5] 	
print('off_data:pos',len(Xpos_train_1),len(Xpos_val_1))
indices=np.arange(len(Xnet_1))
np.random.shuffle(indices)
Xnet_train_1 = Xnet_1[:len(indices)/5] 
Xnet_val_1 = Xnet_1[len(indices)/5:] 
print('off_data:net',len(Xnet_train_1),len(Xnet_val_1))


###### off-target data #######		
input_var_off = T.tensor4('inputs')
target_var_off = T.ivector('targets')
network_off = build_sc_cnn(input_var_off)
prediction_off = lasagne.layers.get_output(network_off)
loss_off = lasagne.objectives.categorical_crossentropy(prediction_off, target_var_off)
loss_off = loss_off.mean()
params_off = lasagne.layers.get_all_params(network_off, trainable=True)
updates_off = lasagne.updates.nesterov_momentum(loss_off, params_off, learning_rate=0.01, momentum=0.9)
train_fn_off = theano.function([input_var_off, target_var_off], loss_off, updates=updates_off)
pre_fn_off = theano.function([input_var_off],[prediction_off])
params_value_off=lasagne.layers.get_all_param_values(network_off)



globeloc=os.getcwd()

fr=open('%s/off.target.sc.train.progress.txt'%globeloc,'w')
fr.write('epochs')
fr.write('\t')
fr.write('training loss')
fr.write('\t')
fr.write('off:acc')
fr.write('\t')
fr.write('off:pre')
fr.write('\t')
fr.write('off:tpr')
fr.write('\t')
fr.write('off:tnr')
fr.write('\t')
fr.write('off:F1')
fr.write('\n')

num_epochs=20
size=50
for epoch in range(num_epochs):

###### train part.1 #########
	train_err = 0
	train_batches = 0
	start_time = time.time()
		
	idn=0
	while idn < 100:
		trainX,trainy=pos_net_pair(Xpos_train_1,Xnet_train_1,size)
		inputs = trainX
		targets = trainy
		train_err += train_fn_off(inputs, targets)
		train_batches += 1
		idn+=1

###### val part #########

	idn=0
	tacc_off=0
	tpre_off=0
	ttpr_off=0
	ttnr_off=0
	tF1_off=0
	tauc_off=0
	
	while idn < 100:
		valX_off,valy_off=pos_net_pair(Xpos_val_1,Xnet_val_1,size)	
		idn+=1

		inputs = valX_off
		targets = valy_off
		prelab=softmax_transf(pre_fn_off(inputs))
		acc,pre,tpr,tnr,F1=Confusion_Matrix(targets,prelab)
		auroc_vr=AUC(pre_fn_off(inputs), targets)
		tacc_off=tacc_off+acc
		tpre_off=tpre_off+pre
		ttpr_off=ttpr_off+tpr
		ttnr_off=ttnr_off+tnr
		tF1_off=tF1_off+F1
		tauc_off=tauc_off+acc

		
	print('')
	print('')
	print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
	print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

	valX_off,valy_off=pos_net_pair(Xpos_val_1,Xnet_val_1,size)	

	inputs = valX_off
	targets = valy_off
	prelab=softmax_transf(pre_fn_off(inputs))
	acc,pre,tpr,tnr,F1=Confusion_Matrix(targets,prelab)
	auroc_vr=AUC(pre_fn_off(inputs), targets)
	print('off-target val set','acc:',acc,'pre:',pre,'tpr:',tpr,'tnr:',tnr,'F1:',F1)
	print('off-target auc:',auroc_vr)
	
	fr.write(str(epoch))
	fr.write('\t')
	fr.write(str(train_err / train_batches))
	fr.write('\t')
	fr.write(str(tacc_off / idn))
	fr.write('\t')
	fr.write(str(tpre_off / idn))
	fr.write('\t')
	fr.write(str(ttpr_off / idn))
	fr.write('\t')
	fr.write(str(ttnr_off / idn))
	fr.write('\t')
	fr.write(str(tF1_off / idn))
	fr.write('\t')
	fr.write(str(tauc_off / idn))
	fr.write('\n')
ISOTIMEFORMAT='%Y-%m-%d'
shijian=time.strftime(ISOTIMEFORMAT,time.localtime())
np.savez('%s.off.hot.sc.model.npz'%shijian, *lasagne.layers.get_all_param_values(network_off))	
	