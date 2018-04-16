import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adagrad, Adam, Adamax, Nadam, RMSprop, Adadelta
from sklearn.metrics import roc_auc_score

#------------------------------------------------------------------------------------------------------------
#documentation says the last 500000 are used as the test set. the data has 11000000 rows total!
data=pd.read_csv("~/Downloads/HIGGS.csv",nrows=1000000,header=None)
test_data=pd.read_csv("~/Downloads/HIGGS.csv",nrows=500000,header=None,skiprows=11000000 - 500000)
# data=pd.read_csv("HIGGS.csv",nrows=1000000,header=None)
# test_data=pd.read_csv("HIGGS.csv",nrows=500000,header=None,skiprows=11000000 - 500000)

y=np.array(data.loc[:,0])
x=np.array(data.loc[:,1:])
x_test=test_data.loc[:,1:]
y_test=test_data.loc[:,0]

def model_test_old(layers, opt_args, mtrcs=['accuracy', 'mae'], epochs= 5, verbose=True):
	#layers is a list of lists where each inner list defines a dens layer
	#	for layer in layers: layer[0] is # of nodes in layer, layer[1] is
	#	kernel initializer, layer[2] is activation function
	model = Sequential()
	layer = layers[0]
	model.add(Dense(layer[0], input_dim=x.shape[1], kernel_initializer=layer[1], activation=layer[2]))
	for layer in layers[1:]:
		model.add(Dense(layer[0], kernel_initializer=layer[1], activation=layer[2]))
	model.add(Dense(1, kernel_initializer=layer[1], activation='sigmoid'))

	#for now just have opt be default
	opt= SGD(lr = opt_args[0], decay = opt_args[1], momentum=0.9, nesterov=True)
	model.compile(loss='binary_crossentropy', metrics=mtrcs, optimizer=opt)

	batch_size=100
	model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=2)
	score = model.evaluate(x_test, y_test, batch_size=batch_size)
	#compute and print results
	s = score + [roc_auc_score(y_test,model.predict(x_test))]
	if verbose:
		scoreStr = ["{:>20}: {:>.4}".format("Test Loss",score[0])] + \
			["{:>20}: {:>.4}".format("Test {}".format(x), y) for x, y in zip(mtrcs, score[1:])] + \
			["{:>20}: {:>.4}".format("Test ROCAUC", s[-1])]
		print("\n".join(scoreStr))
	#return results
	return s

def model_test(layers, opt, opt_args=None, mtrcs=['accuracy', 'mae'], epochs= 5, verbose=True):
	#layers is a list of lists where each inner list defines a dens layer
	#	for layer in layers: layer[0] is # of nodes in layer, layer[1] is
	#	kernel initializer, layer[2] is activation function
	model = Sequential()
	layer = layers[0]
	model.add(Dense(layer[0], input_dim=x.shape[1], kernel_initializer=layer[1], activation=layer[2]))
	for layer in layers[1:]:
		model.add(Dense(layer[0], kernel_initializer=layer[1], activation=layer[2]))
	model.add(Dense(1, kernel_initializer=layer[1], activation='sigmoid'))

	#for now just have opt be default
	model.compile(loss='binary_crossentropy', metrics=mtrcs, optimizer=opt)

	batch_size=100
	model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=2)
	score = model.evaluate(x_test, y_test, batch_size=batch_size)
	#compute and print results
	s = score + [roc_auc_score(y_test,model.predict(x_test))]
	if verbose:
		scoreStr = ["{:>20}: {:>.4}".format("Test Loss",score[0])] + \
			["{:>20}: {:>.4}".format("Test {}".format(x), y) for x, y in zip(mtrcs, score[1:])] + \
			["{:>20}: {:>.4}".format("Test ROCAUC", s[-1])]
		print("\n".join(scoreStr))
	#return results
	return s

import matplotlib.pyplot as plt
def vis_sdat(sdat, metric = 'rocauc', xaxis = 'trial'):
	df = pd.DataFrame(sdat, columns = ["loss", "acc", "mae", "rocauc"])
	df = df.reset_index()
	df.columns = ["trial"] + list(df.columns [1:])
	df.loc[:, "trial"] = df.apply(lambda x: x[0] + 1, axis = 1)
	plt.scatter(df[xaxis], df[metric])
	plt.ylim((0,1))
	plt.show()


#---------------------------------------------------------------------------
#try it out
import random
import itertools
layer_list = [[[x*10, "uniform", random.choice(['relu','tanh','sigmoid'])] for x in np.random.randint(4, 10, 3)]]
lrates = [0.01*(3**x) for x in range(6)]
drates = [10*x*(1e-6) for x in range(20)]
opt_args = itertools.product(lrates, drates)
params = [x for x in itertools.product(layer_list, opt_args)]

sdat = []
for i, param in enumerate(params):
	print("Trial {:>3} of {:>3}".format(i, len(params)))
	sdat.append(model_test_old(param[0], param[1]))

#---------------------------------------------------------------------------
#look at results
vis_sdat(sdat)

#---------------------------------------------------------------------------
#the above took about 5 hours and a lot of our results were terrible, lets look at those with rocauc over 0.75
best_mods = [params[i] for i in list(df[df.rocauc>0.75].index)]

#get the model aucroc and decay rate together to see how they trend
bm_dr = np.array([[best_mod[1][0], best_mod[1][1], df.loc[i,'rocauc']] for i, best_mod in zip(list(df[df.rocauc>0.75].index),best_mods)])
plt.plot(bm_dr[:,0], bm_dr[:,1])
plt.show()

#this is what we learned about this situation:
#so decay rates should be between 0 and 0.00018
#and learning rates are either 0.01 or 0.03

#--------------------------------------------------------------------------
#lets try some of the better models from manual exploration
#this layer setup comes from one that had a rocauc of 0.8276
#we'll try the adagrad optimizer though
layer_list = [[[x*10, "uniform", "relu"] for x in [5, 8, 8]]]
lrates = [0.01, 0.03]
drates = list(np.linspace(0, 0.00018, 5))
opt_args = itertools.product(lrates, drates)
params = [x for x in itertools.product(layer_list, opt_args)]

np.random.seed(128)
model_test_ada(layer_list[0], epochs=10)

np.random.seed(128)
model_test(layer_list[0], [0.01, 0.00018], epochs=10)

#---------------------------------------------------------------------------
#try a few different optimizers
opts = [Adagrad(), Adam(), Adamax(), Nadam(), RMSprop(), Adadelta()]
names = "Adagrad, Adam, Adamax, Nadam, RMSprop, Adadelta".split(', ')
for i, opt in enumerate(opts):
	print("-*"*40)
	print("OPTIMIZER: {}".format(names[i]))
	np.random.seed(128)
	model_test(layer_list[0], opt, epochs=10)

#adamax and adadelta were the best but all except Adagrad had above 0.82 ROCAUC,

#---------------------------------------------------------------------------
#try different # of layers now
layer_list = [[[x*10, "uniform", "relu"] for x in [8, 8, 8]],
	[[x*10, "uniform", "relu"] for x in [8, 8, 8, 8]],
	[[x*10, "uniform", "relu"] for x in [8, 8, 8, 8, 8]],
	[[x*10, "uniform", "relu"] for x in [8, 8, 8, 8, 8, 8]]]

sdat = []
opt = Adadelta()
for i, layer in enumerate(layer_list):
	print("-*"*40)
	print("LAYER CONFIG: {}".format(i))
	np.random.seed(128)
	sdat.append(model_test(layer, opt, epochs=10))

vis_sdat(sdat)

#===================================================================
#try different activations and drop the last layer from the last trial
#try different # of layers now
layer_list = [[[80, "uniform", x] for x in ["tanh", "relu", "relu", "relu"]],
	[[80, "uniform", x] for x in ["relu", "tanh", "relu", "relu"]],
	[[80, "uniform", x] for x in ["relu", "relu", "tanh", "relu"]],
	[[80, "uniform", x] for x in ["relu", "relu", "relu", "tanh"]],
	[[80, "uniform", x] for x in ["tanh", "relu", "relu", "tanh"]]]

sdat = []
opt = Adadelta()
for i, layer in enumerate(layer_list):
	print("-*"*40)
	print("LAYER CONFIG: {}".format(i))
	np.random.seed(128)
	sdat.append(model_test(layer, opt, epochs=10))

#the best was the first where tanh was the first activation, second best was with tanh at the end so we added an extra layer at the end with tanh as the first and also as the last activation, this new layer was actually the second best now with ROCAUC of 0.8303

#===================================================================
#try different activations and drop the last layer from the last trial
#try different # of layers now
layer_list = [[[i, "uniform", x] for i, x in zip([50, 80, 80, 80], ["tanh", "relu", "relu", "relu"])],
	[[i, "uniform", x] for i, x in zip([80, 50, 80, 80], ["tanh", "relu", "relu", "relu"])],
	[[i, "uniform", x] for i, x in zip([80, 80, 50, 80], ["tanh", "relu", "relu", "relu"])],
	[[i, "uniform", x] for i, x in zip([80, 80, 80, 50], ["tanh", "relu", "relu", "relu"])],
	[[i, "uniform", x] for i, x in zip([50, 80, 80, 50], ["tanh", "relu", "relu", "relu"])]]

sdat = []
opt = Adadelta()
for i, layer in enumerate(layer_list):
	print("-*"*40)
	print("LAYER CONFIG: {}".format(i))
	np.random.seed(128)
	sdat.append(model_test(layer, opt, epochs=10))

#the best was the first layer where 50 was the first the second best was when 50 was the third, we artificially added one where 50 nodes were in the first and third layers

#===================================================================
#try different kernel initializers
layer_list = [[[i, "VarianceScaling", x] for i, x in zip([50, 80, 80, 80], ["tanh", "relu", "relu", "relu"])], #0.8296
	[[i, "orthogonal", x] for i, x in zip([50, 80, 80, 80], ["tanh", "relu", "relu", "relu"])], #0.8326
	[[i, "lecun_uniform", x] for i, x in zip([50, 80, 80, 80], ["tanh", "relu", "relu", "relu"])], #0.8305
	[[i, "lecun_normal", x] for i, x in zip([50, 80, 80, 80], ["tanh", "relu", "relu", "relu"])], #0.8289
	[[i, "he_normal", x] for i, x in zip([50, 80, 80, 80], ["tanh", "relu", "relu", "relu"])]] #0.8321

sdat = []
opt = Adadelta()
for i, layer in enumerate(layer_list):
	print("-*"*40)
	print("LAYER CONFIG: {}".format(i))
	np.random.seed(128)
	sdat.append(model_test(layer, opt, epochs=10))

#orthogonal was best and then he_normal, however we used the same initializer for every layer and could have picked different ones at each layer, for now we'll stick with one initializer for each model

#===================================================================
layer_list = [[[i, "orthogonal", x] for i, x in zip([40, 80, 80, 80], ["tanh", "relu", "relu", "relu"])],
	[[i, "orthogonal", x] for i, x in zip([30, 80, 80, 80], ["tanh", "relu", "relu", "relu"])],
	[[i, "orthogonal", x] for i, x in zip([20, 80, 80, 80], ["tanh", "relu", "relu", "relu"])],
	[[i, "orthogonal", x] for i, x in zip([10, 80, 80, 80], ["tanh", "relu", "relu", "relu"])],
	[[i, "orthogonal", x] for i, x in zip([20, 50, 80, 80], ["tanh", "relu", "relu", "relu"])], #0.8291
	[[i, "orthogonal", x] for i, x in zip([20, 80, 50, 80], ["tanh", "relu", "relu", "relu"])], #0.8278
	[[i, "orthogonal", x] for i, x in zip([20, 80, 80, 50], ["tanh", "relu", "relu", "relu"])],
	[[i, "orthogonal", x] for i, x in zip([20, 100, 80, 80], ["tanh", "relu", "relu", "relu"])], #0.8314
	[[i, "orthogonal", x] for i, x in zip([20, 80, 100, 80], ["tanh", "relu", "relu", "relu"])], #0.8285
	[[i, "orthogonal", x] for i, x in zip([20, 80, 80, 100], ["tanh", "relu", "relu", "relu"])], #0.8309
	[[i, "orthogonal", x] for i, x in zip([20, 100, 80, 100], ["tanh", "relu", "relu", "relu"])], #0.8309
	[[i, "orthogonal", x] for i, x in zip([20, 100,100,100], ["tanh", "relu", "relu", "relu"])], #0.8334
	[[i, "orthogonal", x] for i, x in zip([20, 100,120,100], ["tanh", "relu", "relu", "relu"])], #0.8327
	[[i, "orthogonal", x] for i, x in zip([20, 100,120,120], ["tanh", "relu", "relu", "relu"])], #0.8315
	[[i, "orthogonal", x] for i, x in zip([20, 100,120,140], ["tanh", "relu", "relu", "relu"])], #0.8326
	[[i, "orthogonal", x] for i, x in zip([20, 100,20,100], ["tanh", "relu", "relu", "relu"])], #0.829
	[[i, "orthogonal", x] for i, x in zip([20, 20,20,20], ["tanh", "relu", "relu", "relu"])], #0.8161
	[[i, "orthogonal", x] for i, x in zip([20, 20,20,100], ["tanh", "relu", "relu", "relu"])], #0.8185
	[[i, "orthogonal", x] for i, x in zip([20, 20,100,100], ["tanh", "relu", "relu", "relu"])], #0.8258
	[[i, "orthogonal", x] for i, x in zip([20, 100,20,20], ["tanh", "relu", "relu", "relu"])], #0.8298
	[[i, "orthogonal", x] for i, x in zip([20, 20,100,20], ["tanh", "relu", "relu", "relu"])], #0.822
	[[i, "orthogonal", x] for i, x in zip([20, 100,100,140], ["tanh", "relu", "relu", "relu"])], #0.8334
	[[i, "orthogonal", x] for i, x in zip([20, 100,100,120], ["tanh", "relu", "relu", "relu"])], #0.8314
	[[i, "orthogonal", x] for i, x in zip([20, 100,100,160], ["tanh", "relu", "relu", "relu"])], #0.8299
	[[i, "orthogonal", x] for i, x in zip([20, 100,100,180], ["tanh", "relu", "relu", "relu"])] #0.8329
	]

sdat = []
opt = Adadelta()
for i, layer in enumerate(layer_list):
	print("-*"*40)
	print("LAYER CONFIG: {}".format(i))
	np.random.seed(128)
	sdat.append(model_test(layer, opt, epochs=10))

#-------------------------------------------------------------------
#just realized we can have a layer without an activation function...
#it is now necessary to see how it works with these configurations
layer_list = [[[i, "orthogonal", x] for i, x in zip([20, 100,100,140], [None, "relu", "relu", "relu"])], #0.836
	[[i, "orthogonal", x] for i, x in zip([20, 100,100,140], ["tanh", None, "relu", "relu"])], #0.8313
	[[i, "orthogonal", x] for i, x in zip([20, 100,100,140], ["tanh", "relu", None, "relu"])], #0.8322
	[[i, "orthogonal", x] for i, x in zip([20, 100,100,140], ["tanh", "relu", "relu", None])], #0.8327
	[[i, "orthogonal", x] for i, x in zip([20, 100,100,140], [None, None, None, None])], #0.6829
	[[i, "orthogonal", x] for i, x in zip([20, 100,100,140], [None, None, None, "relu"])], #0.8188
	[[i, "orthogonal", x] for i, x in zip([20, 100,100,140], [None, "relu", None, "relu"])], #0.8331
	[[i, "orthogonal", x] for i, x in zip([20, 100,100,140], [None, "tanh", None, "relu"])], #0.8306
	[[i, "orthogonal", x] for i, x in zip([20, 100,100,140], ["tanh", None, None, "relu"])], #0.8197
	[[i, "orthogonal", x] for i, x in zip([20, 100,100,140], [None, "tanh", "tanh", "tanh"])], #0.8323
	[[i, "orthogonal", x] for i, x in zip([20, 100,100,140], [None, "tanh", "tanh", "relu"])], #0.8321
	[[i, "orthogonal", x] for i, x in zip([20, 100,100,140], [None, "tanh", "relu", "relu"])] #0.8355
	]

sdat = []
opt = Adadelta()
for i, layer in enumerate(layer_list):
	print("-*"*40)
	print("LAYER CONFIG: {}".format(i))
	np.random.seed(128)
	sdat.append(model_test(layer, opt, epochs=15))


#-------------------------------------------------------------------
#just realized we can have a layer without an activation function...
#it is now necessary to see how it works with these confiurations
layer_list = [[[i, "orthogonal", x] for i, x in zip([28, 100,100,140], [None, "relu", "relu", "relu"])],
	[[i, "orthogonal", x] for i, x in zip([28, 28, 28, 28], [None, "relu", "relu", "relu"])],
	[[i, "orthogonal", x] for i, x in zip([28, 56, 56, 28], [None, "relu", "relu", "relu"])]
	]

sdat = []
opt = Adadelta()
for i, layer in enumerate(layer_list[-1:]):
	print("-*"*40)
	print("LAYER CONFIG: {}".format(i))
	np.random.seed(128)
	sdat.append(model_test(layer, opt, epochs=15))

