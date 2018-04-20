import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adagrad, Adam, Adamax, Nadam, RMSprop, Adadelta
from sklearn.metrics import roc_auc_score
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

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

class modeler:
	def __init__(self, layers, opt, x_test, y_test, x, y, mtrcs=['accuracy', 'mae'], epochs=5, verbose=True, callbacks=None, validation_split=0):
		# layers is a list of lists where each sub list defines a layer
		# the sub lists have number of nodes, initializer, and activation
		self.epochs = epochs
		self.mtrcs = mtrcs
		self.x_test = x_test
		self.y_test = y_test
		self.x = x
		self.y = y
		self.callbacks = callbacks
		self.validation_split = validation_split
		self.opt = opt
		self.is_compiled = False
		self.model = Sequential()
		layer = layers[0]
		self.model.add(Dense(layer[0], input_dim=x.shape[1], kernel_initializer=layer[1], activation=layer[2]))
		for layer in layers[1:]:
			self.model.add(Dense(layer[0], kernel_initializer=layer[1], activation=layer[2]))
		self.model.add(Dense(1, kernel_initializer=layer[1], activation='sigmoid'))

	def compile(self, loss='binary_crossentropy'):
		self.model.compile(loss=loss, metrics=self.mtrcs, optimizer=self.opt)
		self.is_compiled = True

	def train_model(self, batch_size=100, keras_verbose=2, verbose=True):
		if self.is_compiled == False:
			self.compile()

		self.train_history = self.model.fit(self.x, self.y, epochs=self.epochs, batch_size=batch_size, verbose=keras_verbose, validation_split=self.validation_split, callbacks=self.callbacks)
		self.score = self.model.evaluate(self.x_test, self.y_test, batch_size=batch_size)
		self.s = self.score + [roc_auc_score(self.y_test, self.model.predict(self.x_test))]
		if verbose:
			self.printScores()

	def printScores(self):
		scoreStr = ["{:>20}: {:>.4}".format("Test Loss",self.score[0])] + \
			["{:>20}: {:>.4}".format("Test {}".format(x), y) for x, y in zip(self.mtrcs, self.score[1:])] + \
			["{:>20}: {:>.4}".format("Test ROCAUC", self.s[-1])]
		print("\n".join(scoreStr))

def model_test(layers, opt, mtrcs=['accuracy', 'mae'], epochs= 5, verbose=True):
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
	cb = model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=2)
	score = model.evaluate(x_test, y_test, batch_size=batch_size)
	#compute and print results
	s = score + [roc_auc_score(y_test,model.predict(x_test))]
	if verbose:
		scoreStr = ["{:>20}: {:>.4}".format("Test Loss",score[0])] + \
			["{:>20}: {:>.4}".format("Test {}".format(x), y) for x, y in zip(mtrcs, score[1:])] + \
			["{:>20}: {:>.4}".format("Test ROCAUC", s[-1])]
		print("\n".join(scoreStr))
	#return results
	return s, cb

def plot_loss(hist):
	# assumes hist is a dictionary containing the loss recorded at each epoch for a specific configuration of the neural network
	dfs = [pd.DataFrame([[i, x] for i, x in enumerate(hist[key].history['loss'])], columns=['epoch', 'loss']) for key in hist.keys()]
	for key, df in zip(hist.keys(), dfs):
		plt.plot(df.epoch, df.loss, label=key)
	plt.legend()
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
	sdat.append(model_test(layer, opt, epochs=10)[0])

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
	sdat.append(model_test(layer, opt, epochs=10)[0])

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
	sdat.append(model_test(layer, opt, epochs=10)[0])

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
	sdat.append(model_test(layer, opt, epochs=10)[0])

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
	sdat.append(model_test(layer, opt, epochs=10)[0])

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
	sdat.append(model_test(layer, opt, epochs=15)[0])


#-------------------------------------------------------------------
#NOW USING THE MODELER CLASS
#now to revisit the optimizer
layer = [[i, "orthogonal", x] for i, x in zip([20, 100,100,140], [None, "relu", "relu", "relu"])]

mods = {}
#0.8369, 0.8302, 0.8336, 0.8346, 0.8289
lrs = [1.5, 1.55, 1.6, 1.45, 1.4]

for i, options in enumerate(lrs):
	print("-*"*40)
	print("LAYER CONFIG: {}".format(i))
	np.random.seed(128)
	opt = Adadelta(lr = options)
	mods.update({'learn_rate_{}'.format(options): modeler(layer, opt, x_test, y_test, x, y)})
	mods['learn_rate_{}'.format(options)].train_model()


#-------------------------------------------------------------------
#changing decay values
#0.8393, 0.8382, 0.8355, 0.8309
decay = [0.00001, 0.00003, 0.000003, 0.000001]

for i, options in enumerate(decay):
	print("-*"*40)
	print("LAYER CONFIG: {}".format(i))
	np.random.seed(128)
	opt = Adadelta(lr = 1.5, decay = options)
	mods.update({'decay_{}'.format(options): modeler(layer, opt, x_test, y_test, x, y)})
	mods['decay_{}'.format(options)].train_model()

#-------------------------------------------------------------------
#rho
#0.8392, 0.8403, 0.84, 0.8381
rhos = [0.925, 0.975, 0.985, 0.995]

for i, options in enumerate(rhos):
	print("-*"*40)
	print("LAYER CONFIG: {}".format(i))
	np.random.seed(128)
	opt = Adadelta(lr = 1.5, decay = 0.00001, rho=options)
	mods.update({'rho_{}'.format(options): modeler(layer, opt, x_test, y_test, x, y)})
	mods['rho_{}'.format(options)].train_model()

#-------------------------------------------------------------------
#epsilon
#default value for epsilon is 0.0000001
#0.8382, 0.8405, 0.839, 0.827
eps = [0.0000003, 0.00000003, 0.00000001, 0.000000003]

for i, options in enumerate(eps):
	print("-*"*40)
	print("LAYER CONFIG: {}".format(i))
	np.random.seed(128)
	opt = Adadelta(lr = 1.5, decay = 0.00001, rho=0.975, epsilon=options)
	mods.update({'eps_{}'.format(options): modeler(layer, opt, x_test, y_test, x, y)})
	mods['eps_{}'.format(options)].train_model()


#-------------------------------------------------------------------
#back to activations...
layers = [[[i, "orthogonal", x] for i, x in zip([20, 100,100,140], [None, "sigmoid", "relu", "relu"])], #0.8317
	[[i, "orthogonal", x] for i, x in zip([20, 100,100,140], [None, "relu", "sigmoid", "relu"])], #0.8291
	[[i, "orthogonal", x] for i, x in zip([20, 100,100,140], [None, "relu", "relu", "sigmoid"])], #0.8186
	[[i, "orthogonal", x] for i, x in zip([20, 100,100,140], [None, "tanh", "tanh", "sigmoid"])], #0.8062
	[[i, "orthogonal", x] for i, x in zip([20, 100,100,140], [None, "sigmoid", "tanh", "relu"])], #0.8327
	[[i, "orthogonal", x] for i, x in zip([20, 100,100,140], [None, "sigmoid", "relu", "tanh"])], #0.8178
	[[i, "orthogonal", x] for i, x in zip([20, 100,100,140], [None, "sigmoid", "tanh", "tanh"])] #0.8004
	]

opt = Adadelta(lr = 1.5, decay = 0.00001, rho=0.975, epsilon=3e-8)
for i, layer in enumerate(layers[-3:]):
	print("-*"*40)
	print("LAYER CONFIG: {}".format(i))
	np.random.seed(128)
	mods.update({"config_{:02}".format(i): modeler(layer, opt, x_test, y_test, x, y)})
	mods["config_{:02}".format(i)].train_model()

#-------------------------------------------------------------------
#this is the best setup we have had so far
nn = {'layers': [[i, "orthogonal", x] for i, x in zip([20, 100,100,140], [None, "relu", "relu", "relu"])],
	'optimizer': Adadelta(lr = 1.5, decay = 0.00001, rho=0.975, epsilon=3e-8), 'epochs': 15}

#-------------------------------------------------------------------
#now look at doing a bunch more epochs, lets start with 20 i guess
mdl = modeler(nn['layers'], nn['optimizer'], x_test, y_test, x, y, epochs=50, validation_split=0.1)
mdl.train_model(keras_verbose=1)

#they start to flatten out but not completely, we can probably squeeze into 0.85 ROCAUC
fig = plt.figure()
plt.subplot(121)
plt.plot(mdl.train_history.epoch, mdl.train_history.history['loss'], label='loss')
plt.legend()
plt.subplot(122)
plt.plot(mdl.train_history.epoch, mdl.train_history.history['acc'], label='acc')
plt.legend()
plt.suptitle('Loss and Accuracy by Epoch')
fig.text(.5, -.05, 'Figure 1. Loss and Accuracy curves show some potential\nto increase accuracy with a few more epochs.', ha='center')
plt.show()

mdl = modeler(nn['layers'], nn['optimizer'], x_test, y_test, x, y, epochs=50, validation_split=0.1)
mdl.train_model(keras_verbose=1)

#----------------------------------------------------------------------
layers = [[[300, "orthogonal", x] for x in [None, "relu", "relu", "relu"]],
          [[i, "orthogonal", x] for i, x in zip([20, 100,100,140], [None, "relu", "relu", "relu"])]
    ]

mods = {} #resetting our mods dictionary here
#this is the best optimizer settings as found in the previous exploration
opt = Adadelta(lr = 1.5, decay = 0.00001, rho=0.975, epsilon=3e-8)
es = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=0, mode='auto')
for i, mdl in enumerate(layers):
	print("-*"*40)
	print("LAYER CONFIG: {}".format(i))
	np.random.seed(128)
	mods.update({'config_{}'.format(i): modeler(layer, opt, x_test, y_test, x, y, callbacks=[es],
                                                mtrcs=['accuracy'], validation_split=0.1, epochs=100)})
	mods['config_{}'.format(i)].train_model()

for key in mods.keys():
    print("="*30)
    print(key)
    print(mods[key].printScores())

fig = plt.figure(figsize=(16,8))
plt.subplot(121)
for key in mods.keys():
    plt.plot(mods[key].train_history.epoch, mods[key].train_history.history['loss'],
             label='{}_loss'.format(key))
    if mods[key].train_history.history.get('val_loss', False) != False:
        plt.plot(mods[key].train_history.epoch, mods[key].train_history.history['val_loss'],
                 label='{}_val_loss'.format(key))
plt.legend()
plt.subplot(122)
for key in mods.keys():
    plt.plot(mods[key].train_history.epoch, mods[key].train_history.history['acc'],
             label='{}_acc'.format(key))
    if mods[key].train_history.history.get('val_acc', False) != False:
        plt.plot(mods[key].train_history.epoch, mods[key].train_history.history['val_acc'],
                 label='{}_val_acc'.format(key))
plt.legend()
plt.suptitle('Loss and Accuracy by Epoch')
fig.text(.5, -.05, 
         'Figure 2. Loss and Accuracy curves show some potential\nto increase accuracy with a few more epochs.',
         ha='center')
plt.show()