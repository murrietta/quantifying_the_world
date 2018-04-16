# import keras
import tensorflow as tf
import numpy as np
import pandas as pd

#------------------------------------------------------------------------------------------------------------
#documentation says the last 500000 are used as the test set. the data has 11000000 rows total!
data=pd.read_csv("~/Downloads/HIGGS.csv",nrows=1000000,header=None)
test_data=pd.read_csv("~/Downloads/HIGGS.csv",nrows=500000,header=None,skiprows=11000000 - 500000)

#---------------------
#columns, in case we cared. the first 21 are kinematic properties measured by the particle detectors 
#in the accelerator. the last 7 are derived from the first 21.
cols = "{}{}{}{}".format('lepton pT, lepton eta, lepton phi, missing energy magnitude, missing energy phi, ',
			'jet 1 pt, jet 1 eta, jet 1 phi, jet 1 b-tag, jet 2 pt, jet 2 eta, jet 2 phi, ',
			'jet 2 b-tag, jet 3 pt, jet 3 eta, jet 3 phi, jet 3 b-tag, jet 4 pt, jet 4 eta, ',
			'jet 4 phi, jet 4 b-tag, m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb').split(", ")

#------------------------------------------------------------------------------------------------------------
y=np.array(data.loc[:,0])
x=np.array(data.loc[:,1:])
x_test=test_data.loc[:,1:]
y_test=test_data.loc[:,0]

#------------------------------------------------------------------------------------------------------------
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.metrics import roc_auc_score

#------------------------------------------------------------------------------------------------------------
sdat = []

#============================================================================================================
#001
#11s per epoch
#3s evaluation
       #     Test Loss: 0.5599
       # Test accuracy: 0.7106
       #      Test mae: 0.3808
       #  Test ROCAUC:: 0.7839
np.random.seed(128)
mtrcs = ['accuracy', 'mae']

model = Sequential()
model.add(Dense(50, input_dim=x.shape[1], kernel_initializer='uniform')) # X_train.shape[1] == 15 here
model.add(Activation('relu'))
model.add(Dense(1, kernel_initializer='uniform')) 
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', metrics=mtrcs, optimizer=sgd)


model.fit(x, y, epochs=10, batch_size=100, verbose=2)
score = model.evaluate(x_test, y_test, batch_size=100)

#compute and print results
sdat.append(score + [roc_auc_score(y_test,model.predict(x_test))])
scoreStr = ["{:>20}: {:>.4}".format("Test Loss",score[0])] + \
	["{:>20}: {:>.4}".format("Test {}".format(x), y) for x, y in zip(mtrcs, score[1:])] + \
	["{:>20}: {:>.4}".format("Test ROCAUC", sdat[-1][-1])]
print("\n".join(scoreStr))

#============================================================================================================
#002
#10s per epoch
#3s evaluation
       #     Test Loss: 0.6047
       # Test accuracy: 0.6741
       #      Test mae: 0.4056
       #   Test ROCAUC: 0.747
np.random.seed(128)
mtrcs = ['accuracy', 'mae']

model = Sequential()
model.add(Dense(50, input_dim=x.shape[1], kernel_initializer='uniform')) # X_train.shape[1] == 15 here
model.add(Activation('relu'))
model.add(Dense(1, kernel_initializer='uniform')) 
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.3, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', metrics=mtrcs, optimizer=sgd)


model.fit(x, y, epochs=10, batch_size=100, verbose=2)
score = model.evaluate(x_test, y_test, batch_size=100)

#compute and print results
sdat.append(score + [roc_auc_score(y_test,model.predict(x_test))])
scoreStr = ["{:>20}: {:>.4}".format("Test Loss",score[0])] + \
	["{:>20}: {:>.4}".format("Test {}".format(x), y) for x, y in zip(mtrcs, score[1:])] + \
	["{:>20}: {:>.4}".format("Test ROCAUC", sdat[-1][-1])]
print("\n".join(scoreStr))

#============================================================================================================
#003
#3s per epoch
#3s evaluation
       #     Test Loss: 0.6068
       # Test accuracy: 0.6659
       #      Test mae: 0.4255
       #   Test ROCAUC: 0.733
np.random.seed(128)
mtrcs = ['accuracy', 'mae']

model = Sequential()
model.add(Dense(50, input_dim=x.shape[1], kernel_initializer='uniform')) # X_train.shape[1] == 15 here
model.add(Activation('relu'))
model.add(Dense(1, kernel_initializer='uniform')) 
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.03, decay=1e-6, momentum=0.1, nesterov=True)
model.compile(loss='binary_crossentropy', metrics=mtrcs, optimizer=sgd)


model.fit(x, y, epochs=10, batch_size=1000, verbose=2)
score = model.evaluate(x_test, y_test, batch_size=100)

#compute and print results
sdat.append(score + [roc_auc_score(y_test,model.predict(x_test))])
scoreStr = ["{:>20}: {:>.4}".format("Test Loss",score[0])] + \
	["{:>20}: {:>.4}".format("Test {}".format(x), y) for x, y in zip(mtrcs, score[1:])] + \
	["{:>20}: {:>.4}".format("Test ROCAUC", sdat[-1][-1])]
print("\n".join(scoreStr))

#============================================================================================================
#004
#20s per epoch
#5s evaluation
       #     Test Loss: 0.5607
       # Test accuracy: 0.7059
       #      Test mae: 0.3722
       #   Test ROCAUC: 0.7938
np.random.seed(128)
mtrcs = ['accuracy', 'mae']

model = Sequential()
model.add(Dense(50, input_dim=x.shape[1], kernel_initializer='uniform'))
model.add(Activation('relu'))
model.add(Dense(1, kernel_initializer='uniform')) 
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', metrics=mtrcs, optimizer=sgd)

batch_size = 50
model.fit(x, y, epochs=10, batch_size=batch_size, verbose=2)
score = model.evaluate(x_test, y_test, batch_size=batch_size)

#compute and print results
sdat.append(score + [roc_auc_score(y_test,model.predict(x_test))])
scoreStr = ["{:>20}: {:>.4}".format("Test Loss",score[0])] + \
	["{:>20}: {:>.4}".format("Test {}".format(x), y) for x, y in zip(mtrcs, score[1:])] + \
	["{:>20}: {:>.4}".format("Test ROCAUC", sdat[-1][-1])]
print("\n".join(scoreStr))

#============================================================================================================
#005
#7s per epoch
#2s evaluation
       #     Test Loss: 0.5593
       # Test accuracy: 0.7065
       #      Test mae: 0.3765
       #   Test ROCAUC: 0.788
np.random.seed(128)
mtrcs = ['accuracy', 'mae']

model = Sequential()
model.add(Dense(50, input_dim=x.shape[1], kernel_initializer='uniform'))
model.add(Activation('relu'))
model.add(Dense(1, kernel_initializer='uniform')) 
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', metrics=mtrcs, optimizer=sgd)

batch_size = 150
model.fit(x, y, epochs=10, batch_size=batch_size, verbose=2)
score = model.evaluate(x_test, y_test, batch_size=batch_size)

#compute and print results
sdat.append(score + [roc_auc_score(y_test,model.predict(x_test))])
scoreStr = ["{:>20}: {:>.4}".format("Test Loss",score[0])] + \
	["{:>20}: {:>.4}".format("Test {}".format(x), y) for x, y in zip(mtrcs, score[1:])] + \
	["{:>20}: {:>.4}".format("Test ROCAUC", sdat[-1][-1])]
print("\n".join(scoreStr))

#============================================================================================================
#006
#40s per epoch
#11s evaluation
       #     Test Loss: 0.5565
       # Test accuracy: 0.7098
       #      Test mae: 0.3714
       #   Test ROCAUC: 0.7914
np.random.seed(128)
mtrcs = ['accuracy', 'mae']

model = Sequential()
model.add(Dense(50, input_dim=x.shape[1], kernel_initializer='uniform'))
model.add(Activation('relu'))
model.add(Dense(1, kernel_initializer='uniform')) 
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', metrics=mtrcs, optimizer=sgd)

batch_size = 25
model.fit(x, y, epochs=10, batch_size=batch_size, verbose=2)
score = model.evaluate(x_test, y_test, batch_size=batch_size)

#compute and print results
sdat.append(score + [roc_auc_score(y_test,model.predict(x_test))])
scoreStr = ["{:>20}: {:>.4}".format("Test Loss",score[0])] + \
	["{:>20}: {:>.4}".format("Test {}".format(x), y) for x, y in zip(mtrcs, score[1:])] + \
	["{:>20}: {:>.4}".format("Test ROCAUC", sdat[-1][-1])]
print("\n".join(scoreStr))

#============================================================================================================
#007
#12s per epoch
#2s evaluation
       #     Test Loss: 0.5157
       # Test accuracy: 0.7402
       #      Test mae: 0.3475
       #   Test ROCAUC: 0.8212
np.random.seed(128)
mtrcs = ['accuracy', 'mae']

model = Sequential()
model.add(Dense(50, input_dim=x.shape[1], kernel_initializer='uniform'))
model.add(Activation('relu'))
model.add(Dense(80, kernel_initializer='uniform'))
model.add(Activation('relu'))
model.add(Dense(1, kernel_initializer='uniform')) 
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', metrics=mtrcs, optimizer=sgd)

batch_size = 100
model.fit(x, y, epochs=10, batch_size=batch_size, verbose=2)
score = model.evaluate(x_test, y_test, batch_size=batch_size)

#compute and print results
sdat.append(score + [roc_auc_score(y_test,model.predict(x_test))])
scoreStr = ["{:>20}: {:>.4}".format("Test Loss",score[0])] + \
	["{:>20}: {:>.4}".format("Test {}".format(x), y) for x, y in zip(mtrcs, score[1:])] + \
	["{:>20}: {:>.4}".format("Test ROCAUC", sdat[-1][-1])]
print("\n".join(scoreStr))

#============================================================================================================
#008
#14s per epoch
#2s evaluation
       #     Test Loss: 0.5132
       # Test accuracy: 0.7418
       #      Test mae: 0.3441
       #   Test ROCAUC: 0.8229
np.random.seed(128)
mtrcs = ['accuracy', 'mae']

model = Sequential()
model.add(Dense(50, input_dim=x.shape[1], kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(80, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(80, kernel_initializer='uniform', activation = 'tanh'))
model.add(Dense(1, kernel_initializer='uniform', activation = 'sigmoid')) 

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', metrics=mtrcs, optimizer=sgd)

batch_size = 100
model.fit(x, y, epochs=10, batch_size=batch_size, verbose=2)
score = model.evaluate(x_test, y_test, batch_size=batch_size)

#compute and print results
sdat.append(score + [roc_auc_score(y_test,model.predict(x_test))])
scoreStr = ["{:>20}: {:>.4}".format("Test Loss",score[0])] + \
	["{:>20}: {:>.4}".format("Test {}".format(x), y) for x, y in zip(mtrcs, score[1:])] + \
	["{:>20}: {:>.4}".format("Test ROCAUC", sdat[-1][-1])]
print("\n".join(scoreStr))

#============================================================================================================
#009
#14s per epoch
#4s evaluation
       #     Test Loss: 0.5068
       # Test accuracy: 0.7453
       #      Test mae: 0.3412
       #   Test ROCAUC: 0.8276
np.random.seed(128)
mtrcs = ['accuracy', 'mae']

model = Sequential()
model.add(Dense(50, input_dim=x.shape[1], kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(80, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(80, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(1, kernel_initializer='uniform', activation = 'sigmoid')) 

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', metrics=mtrcs, optimizer=sgd)

batch_size = 100
model.fit(x, y, epochs=10, batch_size=batch_size, verbose=2)
score = model.evaluate(x_test, y_test, batch_size=batch_size)

#compute and print results
sdat.append(score + [roc_auc_score(y_test,model.predict(x_test))])
scoreStr = ["{:>20}: {:>.4}".format("Test Loss",score[0])] + \
	["{:>20}: {:>.4}".format("Test {}".format(x), y) for x, y in zip(mtrcs, score[1:])] + \
	["{:>20}: {:>.4}".format("Test ROCAUC", sdat[-1][-1])]
print("\n".join(scoreStr))

#============================================================================================================
#010
#20s per epoch
#4s evaluation
       #     Test Loss: 0.511
       # Test accuracy: 0.743
       #      Test mae: 0.3434
       #   Test ROCAUC: 0.8247
np.random.seed(128)
mtrcs = ['accuracy', 'mae']

model = Sequential()
model.add(Dense(50, input_dim=x.shape[1], kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(80, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(80, kernel_initializer='uniform', activation = 'selu'))
model.add(Dense(1, kernel_initializer='uniform', activation = 'sigmoid')) 

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', metrics=mtrcs, optimizer=sgd)

batch_size = 100
model.fit(x, y, epochs=10, batch_size=batch_size, verbose=2)
score = model.evaluate(x_test, y_test, batch_size=batch_size)

#compute and print results
sdat.append(score + [roc_auc_score(y_test,model.predict(x_test))])
scoreStr = ["{:>20}: {:>.4}".format("Test Loss",score[0])] + \
	["{:>20}: {:>.4}".format("Test {}".format(x), y) for x, y in zip(mtrcs, score[1:])] + \
	["{:>20}: {:>.4}".format("Test ROCAUC", sdat[-1][-1])]
print("\n".join(scoreStr))

#============================================================================================================
#011
#14s per epoch
#4s evaluation
       #     Test Loss: 0.6915
       # Test accuracy: 0.529
       #      Test mae: 0.4983
       #   Test ROCAUC: 0.4999
np.random.seed(128)
mtrcs = ['accuracy', 'mae']

model = Sequential()
model.add(Dense(50, input_dim=x.shape[1], kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(80, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(1, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(80, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(1, kernel_initializer='uniform', activation = 'sigmoid')) 

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', metrics=mtrcs, optimizer=sgd)

batch_size = 100
model.fit(x, y, epochs=10, batch_size=batch_size, verbose=2)
score = model.evaluate(x_test, y_test, batch_size=batch_size)

#compute and print results
sdat.append(score + [roc_auc_score(y_test,model.predict(x_test))])
scoreStr = ["{:>20}: {:>.4}".format("Test Loss",score[0])] + \
	["{:>20}: {:>.4}".format("Test {}".format(x), y) for x, y in zip(mtrcs, score[1:])] + \
	["{:>20}: {:>.4}".format("Test ROCAUC", sdat[-1][-1])]
print("\n".join(scoreStr))

#============================================================================================================
#012
#17s per epoch
#4s evaluation
       #     Test Loss: 0.5093
       # Test accuracy: 0.7439
       #      Test mae: 0.3412
       #   Test ROCAUC: 0.826
np.random.seed(128)
mtrcs = ['accuracy', 'mae']

model = Sequential()
model.add(Dense(50, input_dim=x.shape[1], kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(80, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(50, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(80, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(1, kernel_initializer='uniform', activation = 'sigmoid')) 

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', metrics=mtrcs, optimizer=sgd)

batch_size = 100
model.fit(x, y, epochs=10, batch_size=batch_size, verbose=2)
score = model.evaluate(x_test, y_test, batch_size=batch_size)

#compute and print results
sdat.append(score + [roc_auc_score(y_test,model.predict(x_test))])
scoreStr = ["{:>20}: {:>.4}".format("Test Loss",score[0])] + \
	["{:>20}: {:>.4}".format("Test {}".format(x), y) for x, y in zip(mtrcs, score[1:])] + \
	["{:>20}: {:>.4}".format("Test ROCAUC", sdat[-1][-1])]
print("\n".join(scoreStr))

#============================================================================================================
#013
#21s per epoch
#5s evaluation
       #     Test Loss: 0.6916
       # Test accuracy: 0.529
       #      Test mae: 0.4979
       #   Test ROCAUC: 0.5
np.random.seed(128)
mtrcs = ['accuracy', 'mae']

model = Sequential()
model.add(Dense(50, input_dim=x.shape[1], kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(80, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(50, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(80, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(50, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(80, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(50, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(80, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(50, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(1, kernel_initializer='uniform', activation = 'sigmoid')) 

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', metrics=mtrcs, optimizer=sgd)

batch_size = 100
model.fit(x, y, epochs=10, batch_size=batch_size, verbose=2)
score = model.evaluate(x_test, y_test, batch_size=batch_size)

#compute and print results
sdat.append(score + [roc_auc_score(y_test,model.predict(x_test))])
scoreStr = ["{:>20}: {:>.4}".format("Test Loss",score[0])] + \
	["{:>20}: {:>.4}".format("Test {}".format(x), y) for x, y in zip(mtrcs, score[1:])] + \
	["{:>20}: {:>.4}".format("Test ROCAUC", sdat[-1][-1])]
print("\n".join(scoreStr))

#============================================================================================================
#014
#17s per epoch
#4s evaluation
       #     Test Loss: 0.5403
       # Test accuracy: 0.7225
       #      Test mae: 0.3598
       #   Test ROCAUC: 0.8013
np.random.seed(128)
mtrcs = ['accuracy', 'mae']

model = Sequential()
model.add(Dense(50, input_dim=x.shape[1], kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(80, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(50, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(80, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(80, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(1, kernel_initializer='uniform', activation = 'sigmoid')) 

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', metrics=mtrcs, optimizer=sgd)

batch_size = 100
model.fit(x, y, epochs=10, batch_size=batch_size, verbose=2)
score = model.evaluate(x_test, y_test, batch_size=batch_size)

#compute and print results
sdat.append(score + [roc_auc_score(y_test,model.predict(x_test))])
scoreStr = ["{:>20}: {:>.4}".format("Test Loss",score[0])] + \
	["{:>20}: {:>.4}".format("Test {}".format(x), y) for x, y in zip(mtrcs, score[1:])] + \
	["{:>20}: {:>.4}".format("Test ROCAUC", sdat[-1][-1])]
print("\n".join(scoreStr))

#============================================================================================================
#015
#18s per epoch
#4s evaluation
       #     Test Loss: 0.5362
       # Test accuracy: 0.7251
       #      Test mae: 0.3578
       #   Test ROCAUC: 0.8048
np.random.seed(128)
mtrcs = ['accuracy', 'mae']

model = Sequential()
model.add(Dense(50, input_dim=x.shape[1], kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(80, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(80, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(80, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(30, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(1, kernel_initializer='uniform', activation = 'sigmoid')) 

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', metrics=mtrcs, optimizer=sgd)

batch_size = 100
model.fit(x, y, epochs=10, batch_size=batch_size, verbose=2)
score = model.evaluate(x_test, y_test, batch_size=batch_size)

#compute and print results
sdat.append(score + [roc_auc_score(y_test,model.predict(x_test))])
scoreStr = ["{:>20}: {:>.4}".format("Test Loss",score[0])] + \
	["{:>20}: {:>.4}".format("Test {}".format(x), y) for x, y in zip(mtrcs, score[1:])] + \
	["{:>20}: {:>.4}".format("Test ROCAUC", sdat[-1][-1])]
print("\n".join(scoreStr))

#============================================================================================================
#016
#17s per epoch
#4s evaluation
       #     Test Loss: 0.5163
       # Test accuracy: 0.7392
       #      Test mae: 0.3411
       #   Test ROCAUC: 0.8235
np.random.seed(128)
mtrcs = ['accuracy', 'mae']

model = Sequential()
model.add(Dense(50, input_dim=x.shape[1], kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(80, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(90, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(100, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(110, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(1, kernel_initializer='uniform', activation = 'sigmoid')) 

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', metrics=mtrcs, optimizer=sgd)

batch_size = 100
model.fit(x, y, epochs=10, batch_size=batch_size, verbose=2)
score = model.evaluate(x_test, y_test, batch_size=batch_size)

#compute and print results
sdat.append(score + [roc_auc_score(y_test,model.predict(x_test))])
scoreStr = ["{:>20}: {:>.4}".format("Test Loss",score[0])] + \
	["{:>20}: {:>.4}".format("Test {}".format(x), y) for x, y in zip(mtrcs, score[1:])] + \
	["{:>20}: {:>.4}".format("Test ROCAUC", sdat[-1][-1])]
print("\n".join(scoreStr))


#============================================================================================================
#017
#17s per epoch
#4s evaluation
       #     Test Loss: 0.6915
       # Test accuracy: 0.529
       #      Test mae: 0.4984
       #   Test ROCAUC: 0.5502
np.random.seed(128)
mtrcs = ['accuracy', 'mae']

model = Sequential()
model.add(Dense(50, input_dim=x.shape[1], kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(80, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(90, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(100, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(110, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(100, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(90, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(80, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(50, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(1, kernel_initializer='uniform', activation = 'sigmoid')) 

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', metrics=mtrcs, optimizer=sgd)

batch_size = 100
model.fit(x, y, epochs=10, batch_size=batch_size, verbose=2)
score = model.evaluate(x_test, y_test, batch_size=batch_size)

#compute and print results
sdat.append(score + [roc_auc_score(y_test,model.predict(x_test))])
scoreStr = ["{:>20}: {:>.4}".format("Test Loss",score[0])] + \
	["{:>20}: {:>.4}".format("Test {}".format(x), y) for x, y in zip(mtrcs, score[1:])] + \
	["{:>20}: {:>.4}".format("Test ROCAUC", sdat[-1][-1])]
print("\n".join(scoreStr))


#============================================================================================================
#018
#17s per epoch
#4s evaluation
       #     Test Loss: 0.5403
       # Test accuracy: 0.7226
       #      Test mae: 0.3611
       #   Test ROCAUC: 0.8028       
np.random.seed(128)
mtrcs = ['accuracy', 'mae']

model = Sequential()
model.add(Dense(50, input_dim=x.shape[1], kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(50, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(50, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(50, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(50, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(1, kernel_initializer='uniform', activation = 'sigmoid')) 

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', metrics=mtrcs, optimizer=sgd)

batch_size = 100
model.fit(x, y, epochs=10, batch_size=batch_size, verbose=2)
score = model.evaluate(x_test, y_test, batch_size=batch_size)

#compute and print results
sdat.append(score + [roc_auc_score(y_test,model.predict(x_test))])
scoreStr = ["{:>20}: {:>.4}".format("Test Loss",score[0])] + \
	["{:>20}: {:>.4}".format("Test {}".format(x), y) for x, y in zip(mtrcs, score[1:])] + \
	["{:>20}: {:>.4}".format("Test ROCAUC", sdat[-1][-1])]
print("\n".join(scoreStr))


#============================================================================================================
#019
#17s per epoch
#4s evaluation
       #     Test Loss: 0.5529
       # Test accuracy: 0.713
       #      Test mae: 0.3743
       #   Test ROCAUC: 0.7904
np.random.seed(128)
mtrcs = ['accuracy', 'mae']

model = Sequential()
model.add(Dense(50, input_dim=x.shape[1], kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(80, kernel_initializer='uniform', activation = 'tanh'))
model.add(Dense(80, kernel_initializer='uniform', activation = 'tanh'))
model.add(Dense(80, kernel_initializer='uniform', activation = 'tanh'))
model.add(Dense(30, kernel_initializer='uniform', activation = 'tanh'))
model.add(Dense(1, kernel_initializer='uniform', activation = 'sigmoid')) 

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', metrics=mtrcs, optimizer=sgd)

batch_size = 100
model.fit(x, y, epochs=10, batch_size=batch_size, verbose=2)
score = model.evaluate(x_test, y_test, batch_size=batch_size)

#compute and print results
sdat.append(score + [roc_auc_score(y_test,model.predict(x_test))])
scoreStr = ["{:>20}: {:>.4}".format("Test Loss",score[0])] + \
	["{:>20}: {:>.4}".format("Test {}".format(x), y) for x, y in zip(mtrcs, score[1:])] + \
	["{:>20}: {:>.4}".format("Test ROCAUC", sdat[-1][-1])]
print("\n".join(scoreStr))


def model_test(layers, opt_args, mtrcs=['accuracy', 'mae']):
	#layers is a list of lists where each inner list defines a dens layer
	#	for layer in layers: layer[0] is # of nodes in layer, layer[1] is
	#	kernel initializer, layer[2] is activation function
	model = Sequential()
	layer = layers[0]
	model.add(Dense(layer[0], input_dim=x.shape[1], kernel_initializer=layer[1], activation=layer[2]))
	for layer in layers[1:]:
		model.add(Dense(layer[0], kernel_initializer=layer[1], activation=layer[2]))

	opt = SGD(lr=opt_args[0], decay=opt_args[1], momentum=0.9, nesterov=True)
	model.compile(loss='binary_crossentropy', metrics=mtrcs, optimizer=opt)

	batch_size=100
	score = model.evaluate(x_test, y_test, batch_size=batch_size)

	#compute and print results
	return score + [roc_auc_score(y_test,model.predict(x_test))]
	# scoreStr = ["{:>20}: {:>.4}".format("Test Loss",score[0])] + \
	# 	["{:>20}: {:>.4}".format("Test {}".format(x), y) for x, y in zip(mtrcs, score[1:])] + \
	# 	["{:>20}: {:>.4}".format("Test ROCAUC", sdat[-1][-1])]
	# print("\n".join(scoreStr))

#try it out
layer_list = [[[50, "uniform", "relu"],
				[50, "uniform", "relu"],
				[50, "uniform", "relu"],
				[50, "uniform", "relu"],
				[50, "uniform", "relu"],
				[50, "uniform", "relu"],
				[50, "uniform", "relu"],
				[50, "uniform", "relu"],
				[1, "uniform", "sigmoid"]],
				[[50, "uniform", "relu"],
				[100, "uniform", "relu"],
				[100, "uniform", "relu"],
				[100, "uniform", "relu"],
				[100, "uniform", "relu"],
				[100, "uniform", "relu"],
				[100, "uniform", "relu"],
				[100, "uniform", "relu"],
				[1, "uniform", "sigmoid"]],
				[[50, "uniform", "relu"],
				[100, "uniform", "relu"],
				[100, "uniform", "relu"],
				[100, "uniform", "relu"],
				[100, "uniform", "relu"],
				[100, "uniform", "relu"],
				[100, "uniform", "relu"],
				[50, "uniform", "relu"],
				[1, "uniform", "sigmoid"]],
				[[50, "uniform", "relu"],
				[100, "uniform", "relu"],
				[100, "uniform", "relu"],
				[100, "uniform", "relu"],
				[100, "uniform", "relu"],
				[100, "uniform", "relu"],
				[75, "uniform", "relu"],
				[50, "uniform", "relu"],
				[1, "uniform", "sigmoid"]]]
import itertools
lrates = [0.001*(3**x) for x in range(9)]
drates = [10*x*(1e-6) for x in range(20)]
opt_args = itertools.product(lrates, drates)
params = [x for x in itertools.product(layer_list, opt_args)]

sdat = []
for param in params:
	sdat.append(model_test(param[0], param[1]))

#---------------------------------------------------------
#here's all the results so far
[[0.5599, 0.7106, 0.3808, 0.7839],
 [0.6047, 0.6741, 0.4056, 0.747],
 [0.6068, 0.6659, 0.4255, 0.733],
 [0.5607, 0.7059, 0.3722, 0.7938],
 [0.5593463147073984,
  0.7065440012931824,
  0.3765046732336283,
  0.7879831349168613],
 [0.556493350084126,
  0.7097500039130449,
  0.3713838232755661,
  0.7913770794444449],
 [0.5156913005769252,
  0.7402060006380081,
  0.34748699160814284,
  0.8211798930764069],
 [0.513178469389677,
  0.7418480000734329,
  0.3441325005412102,
  0.8229274348603537],
 [0.5068483791768551,
  0.745348000228405,
  0.3411563220500946,
  0.8276033498614552],
 [0.5109848039865493,
  0.7429519998669625,
  0.34335046809315684,
  0.8247020979765738],
 [0.691462618470192,
  0.5290139969229698,
  0.4983312557220459,
  0.49988209576310166],
 [0.5093061820983886,
  0.7438600002765655,
  0.3411905131161213,
  0.8260157984769865],
 [0.6915550626516342, 0.5290139969229698, 0.49792276188731194, 0.5],
 [0.5402580774962902,
  0.7225180010080338,
  0.35980399896502496,
  0.8012546522173533],
 [0.5361666767597199,
  0.7251220012307167,
  0.3578343632102013,
  0.8047882369803495],
 [0.5163418616116047,
  0.7392000003933906,
  0.341137904548645,
  0.8235132492752298],
 [0.6915550538539886, 0.5290139969229698, 0.497922740572691, 0.5],
 [0.6914631579875946,
  0.5290139969229698,
  0.4983684213221073,
  0.5501617104625584],
 [0.5402762879431248,
  0.7226040016174317,
  0.3611249549269676,
  0.8027638026222189],
 [0.5529272284209729,
  0.712998001730442,
  0.3743128134548664,
  0.790448764456932]]

import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame(sdat, columns = ["loss", "acc", "mae", "rocauc"])
df = df.reset_index()
df.columns = ["trial"] + list(df.columns [1:])
df.loc[:, "trial"] = df.apply(lambda x: x[0] + 1, axis = 1)
plt.scatter(df.trial, df.rocauc)
plt.ylim((0,1))
plt.show()