# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 15:32:42 2021

@author: user
"""
import numpy as np
import autokeras as ak
import tensorflow as tf

tf.keras.backend.set_floatx('float64')


t = np.linspace(1/4, 7/4, 102)

p_4_dat=np.genfromtxt("wolfram_sln/p_IV_sln_101.csv",delimiter=',')


from sklearn.model_selection import train_test_split

t_dat=t.reshape(-1,1)

t_dat=np.array(t_dat,dtype=np.float64)

dat=p_4_dat.reshape(-1)

dat=np.array(dat,dtype=np.float64)

x_train,x_test,y_train,y_test = train_test_split(t_dat,dat,test_size = 0.2)

train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))


model = ak.StructuredDataRegressor(max_trials=10, overwrite=True)

model.fit(train_set, epochs=100)

test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test))

print(model.evaluate(test_set))
    
model = model.export_model()