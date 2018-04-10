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
model.add(Dense(50, input_dim=x.shape[1], kernel_initializer='uniform')) # X_train.shape[1] == 15 here
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
#005 NOT DONE!
#20s per epoch
#5s evaluation
       #     Test Loss: 0.5607
       # Test accuracy: 0.7059
       #      Test mae: 0.3722
       #   Test ROCAUC: 0.7938
np.random.seed(128)
mtrcs = ['accuracy', 'mae']

model = Sequential()
model.add(Dense(50, input_dim=x.shape[1], kernel_initializer='uniform')) # X_train.shape[1] == 15 here
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







#------------------------------------------------------------------------------------------------------------
np.random.seed(128)
model = Sequential()
model.add(Dense(50, input_dim=x.shape[1], kernel_initializer='uniform')) # X_train.shape[1] == 15 here
model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(150, kernel_initializer='uniform'))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(50, kernel_initializer='uniform'))
# model.add(Activation('relu'))
model.add(Dense(1, kernel_initializer='uniform')) 
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', metrics=['accuracy', 'mae'], optimizer=sgd)

#------------------------------------------------------------------------------------------------------------
model.fit(x, y, epochs=10, batch_size=100, verbose=2)
#score = model.evaluate(x_test, y_test, batch_size=100)
roc_auc_score(y_test,model.predict(x_test))

#------------------------------------------------------------------------------------------------------------
# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
            
#------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------