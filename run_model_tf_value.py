#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 21:37:00 2018

@author: elizabeth liang
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split


#import keras.backend as K

#from keras.callbacks import Callback
#from keras.layers import Dense
#from keras.layers.recurrent import LSTM
#from keras.models import Sequential
#from keras.optimizers import Adam



import tensorflow as tf

from data_reader import z_score_inv
from next_batch import LSTM_WINDOW_SIZE, INPUT_SIZE, PREDICTORS
from next_batch import get_trainable_data

from data_reader import read_vxx


##############################
seed=4

##############################

# set DISPLAY=0;

plt.ion()

##################################################################################################
##'data.npz' saves y variables as value of volatility
##'data_binary.npz' saves y variables as 0(decrease) and 1(increase)
DATA_FILE = 'data.npz'
#DATA_FILE = 'data_binary.npz'
if not os.path.exists(DATA_FILE):
    (x_train, y_train), (x_test, y_test), mean, std = get_trainable_data()
    np.savez_compressed(DATA_FILE, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                        mean=mean, std=std)
else:
    d = np.load(DATA_FILE)
    x_train = d['x_train']
    y_train = d['y_train']
    x_test = d['x_test']
    y_test = d['y_test']
    mean = d['mean']
    std = d['std']


###################################

x_train = x_train[:,:,[0,1,2]]
x_test = x_test[:,:,[0,1,2]]

###################################

print('x_train.shape =', x_train.shape)
print('y_train.shape =', y_train.shape)

print('x_test.shape =', x_test.shape)
print('y_test.shape =', y_test.shape)

#train_test_split correct?????????
x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.2,random_state=seed)

##################################

print('x_train.shape =',x_train.shape)
print('x_val.shape =',x_val.shape)


# Define the model function (following TF Estimator Template)
# Here the class remain the same for all the structures
class TFClassifier:
    def __init__(self,
                input_size, 
                window_size,
                logit_function,
                max_iter=600,
                learning_rate=0.0001,
                batch_size=32,
                model_dir="./tfclassifier",
                load_from_disk=False,
                summary_steps=None,
                dtype=tf.float32,
                use_adam_optimizer=False,
                X_val=None,
                Y_val=None):

        
        self.learning_rate=learning_rate
        self.max_iter=max_iter
        self.batch_size=batch_size
        self.dtype=dtype
        self.use_adam_optimizer=use_adam_optimizer
        self.g=tf.Graph()
        
        with self.g.as_default():           
            self.build_model(input_size,window_size,logit_function)
        
        self.load_from_disk=load_from_disk
        if not self.load_from_disk:
            shutil.rmtree(model_dir,ignore_errors=True)
        self.dir=model_dir

        if summary_steps is None:
            self.summary_steps=max_iter//20
        else:
            self.summary_steps=summary_steps
   
        self.X_val=X_val
        self.Y_val=Y_val
        
        self.perms=[]
        
        

    #Define loss function and optimizer          
    def build_model(self,input_size,window_size,logit_function):
    
        self.global_step = tf.Variable(0,name='global_step')
        self.in_training=tf.placeholder_with_default(0.0,shape=(None),name="in_training")
        
        self.X=tf.placeholder(self.dtype,[None,window_size,input_size],name="X")
        self.Y=tf.placeholder(self.dtype,[None,1],name="Y")
   
        # Predictions
        self.pred=logit_function(self.X,self.in_training)
        
        # Define loss and optimizer
        # If we want to predict value, use MAPE;
        # If we want to predict direction, use cross entropy.
        self.loss = tf.reduce_mean(tf.abs(tf.divide(tf.subtract(self.pred,self.Y),self.Y)))
    
        if self.use_adam_optimizer:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.training_op = optimizer.minimize(self.loss, global_step=self.global_step)
    
        tf.summary.scalar("loss",self.loss)
        self.summary=tf.summary.merge_all()
        
        self.init = tf.global_variables_initializer()
        
        return 

    #Report value of loss function over a series of epochs
    def report_summary(self,X,Y,sess,epoch,saver,train_summary_writer,val_summary_writer):
        # we only want to report on a small say 1000, subset or train and valdiation
        # sets, if not , reporting will be more expensive than training
        N=min(1000,X.shape[0])
        perm=np.random.choice(X.shape[0],N)
        self.last_checkpoint=saver.save(sess,self.dir+"/optimization.ckpt",global_step=epoch)
        xu=X[perm]
        yu=Y[perm]
        train_loss,train_summary=sess.run(fetches=[self.loss,self.summary],
            feed_dict={self.X:xu,
                       self.Y:yu})  
        train_summary_writer.add_summary(train_summary,epoch)
        
        N=min(1000,self.X_val.shape[0])
        perm=np.random.choice(self.X_val.shape[0],N)
        val_loss,val_summary=sess.run(fetches=[self.loss,self.summary],
            feed_dict={self.X:self.X_val[perm],
                       self.Y:self.Y_val[perm]}
                       )  
        val_summary_writer.add_summary(val_summary,epoch)
        print("Epoch", epoch, "Loss =",train_loss,"Evaluation Loss =",val_loss)
    
    #Train Neural Network
    def fit(self,X,Y):
        N=len(X)
        
        with tf.Session(graph=self.g) as sess:
            #tf.set_random_seed(3)
            
            saver = tf.train.Saver()
            if self.load_from_disk:
                saver.restore(sess,tf.train.latest_checkpoint(self.dir))
            else:
                if not os.path.exists(self.dir):
                    os.mkdir(self.dir)
                sess.run(self.init)
            
            step0=tf.train.global_step(sess,self.global_step)
            train_summary_writer = tf.summary.FileWriter(self.dir+"/train", self.g)
            val_summary_writer = tf.summary.FileWriter(self.dir+"/validation", self.g)
            step=step0
            step_last=step
            
            #set random seed
            np.random.seed(seed)
            while step<step0+self.max_iter:  
                if  (step% self.summary_steps)<=(step_last % self.summary_steps): # we have grapped around
                   self.report_summary(X,Y,sess,step,saver,train_summary_writer,val_summary_writer)
                step_last=step
                
                ##################################
                #need to set seed!!!!!!!!!!
                perm=np.random.permutation(N)
                
                self.perms.append(perm)
                
                ##################################
                for i in range(0,N,self.batch_size):
                    Xb=X[perm[i:i+self.batch_size]]
                    Yb=Y[perm[i:i+self.batch_size]]
                    sess.run([self.training_op],feed_dict={self.X:Xb,
                                                           self.Y:Yb,
                                                           self.in_training:1.0
                                                           }) 
                    
                    step+=1
            self.report_summary(X,Y,sess,step0+self.max_iter,saver,train_summary_writer,val_summary_writer)
            self.load_from_disk=True # make sure we restart next fit from previous level so that we can call fit multiple times

    #Predict y variables
    def predict(self,X,batch_size=1000):
        
        with tf.Session(graph=self.g) as sess:
            saver = tf.train.Saver()
            saver.restore(sess,tf.train.latest_checkpoint(self.dir))
            ys=[]
            for start in range(0,X.shape[0],batch_size):
                
                yb=sess.run([self.pred],feed_dict={self.X:X[slice(start,start+batch_size)]})
                ys.append(yb)
        y=np.concatenate(ys,axis=1).ravel()
        return y

#Definition of RNN structure
#We modify architecture to reduce parameters and improve robustness in this class
class LSTM_Model:
    def __init__(self,input_size = INPUT_SIZE,window_size = LSTM_WINDOW_SIZE,hidden1 = 32,hidden2 = 16):
        self.input_size=input_size
        self.window_size=window_size
        self.hidden1=hidden1
        self.hidden2=hidden2
        
    def __call__(self,x,in_training):

        #x = tf.placeholder(tf.float64, [self.window_size,self.input_size], name='input_placeholder')
        #y = tf.placeholder(tf.float64, [1, 1], name='output_placeholder')   #shape=(None)

        #tf.Variable??????????????
        #x = tf.Variable(tf.float64, [self.window_size,self.input_size], name='input_placeholder')
        #y = tf.placeholder(tf.float64, [1, 1], name='output_placeholder')   #shape=(None)
       
        '''
        cell1 =tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.hidden1,state_is_tuple = True,
                                                                     initializer=tf.glorot_uniform_initializer(seed=seed)),
                                                                     input_keep_prob=1.0-0.0*in_training # no dropout for first layer
                                                                     )
            
        ''' 
        #GRU cell
        cell1 =tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(self.hidden1,
                                                                    activation=tf.nn.tanh,
                                                                    kernel_initializer=tf.glorot_uniform_initializer(seed=seed)
                                                                    ),
                                                                    input_keep_prob=1.0-0.0*in_training # no dropout for first layer
                                                                    )
            
        #        cell1 =tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.hidden1,state_is_tuple = True),
        #                                                                     input_keep_prob=1.0-0.0*in_training # no dropout for first layer
        #                                                                     )

        #cell2 = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.hidden2,state_is_tuple = True),input_keep_prob=1.0-0.4*in_training)
        
        
        #cell = tf.nn.rnn_cell.MultiRNNCell([cell1,cell2]) 


        output, state = tf.nn.dynamic_rnn(cell1, 
                                      x,
                                      #initial_state=initial_state, 
                                      dtype = tf.float32)
        
        #output = tf.transpose(output, [1, 0, 2])
        #print("output",output.shape)
        #last = tf.gather(output, int(output.get_shape()[0]) - 1)
        #print("last",last.shape)
        
        
        #        output1=tf.layers.dense(inputs=state.h, units=self.hidden2,activation='sigmoid')
        #        output2=tf.layers.dense(inputs=output1, units=1)
       
        
        output1=tf.layers.dense(inputs=state, units=self.hidden2,activation='sigmoid',
                                kernel_initializer=tf.glorot_uniform_initializer(seed=seed)
                                )
        '''
        output1=tf.layers.dense(inputs=state.h, units=self.hidden2,activation='sigmoid',kernel_initializer=tf.glorot_uniform_initializer(seed=seed))
        '''
        output2=tf.layers.dense(inputs=output1, units=1,kernel_initializer=tf.glorot_uniform_initializer(seed=seed))
       

        print(x.shape)
        #print(state.h.shape)
        print(output1.shape)
        print(output2.shape)
        return output2



INPUT_SIZE = x_train.shape[2]
lstm_model=LSTM_Model(INPUT_SIZE,LSTM_WINDOW_SIZE,32,16)
#lstm_model=LSTM_Model(INPUT_SIZE,LSTM_WINDOW_SIZE,16,8)

# Build the Estimator
model = TFClassifier(INPUT_SIZE,
                     LSTM_WINDOW_SIZE,
                     lstm_model,
                     max_iter=20000,
                     #max_iter=30000,
                     learning_rate=0.001,
                     batch_size=32,
                     model_dir="./tfclassifier",
                     load_from_disk=False,
                     summary_steps=None,
                     dtype=tf.float32,
                     use_adam_optimizer=True,
                     X_val=x_val,
                     Y_val=y_val
                    )


#model = TFClassifier(INPUT_SIZE,
#                     LSTM_WINDOW_SIZE,
#                     lstm_model,
#                     max_iter=50000,
#                     learning_rate=0.0001,
#                     batch_size=32,
#                     model_dir="./tfclassifier",
#                     load_from_disk=False,
#                     summary_steps=None,
#                     dtype=tf.float32,
#                     use_adam_optimizer=True,
#                     X_val=x_val,
#                     Y_val=y_val
#                    )

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

train_pred = model.predict(x_train)

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

pred_sigmas = [z_score_inv(pred, mean, std) for pred in y_pred.flatten()]
true_sigmas = [z_score_inv(true, mean, std) for true in y_test.flatten()]
dummy_sigmas = [z_score_inv(dummy, mean, std) for dummy in np.roll(y_test.flatten(), shift=1)]

train_sigmas = [z_score_inv(pred, mean, std) for pred in train_pred.flatten()]
train_true_sigmas = [z_score_inv(true, mean, std) for true in y_train.flatten()]

train_mape = mean_absolute_percentage_error(np.array(train_true_sigmas), np.array(train_sigmas))
test_mape = mean_absolute_percentage_error(np.array(true_sigmas), np.array(pred_sigmas))
dummy_mape = mean_absolute_percentage_error(np.array(true_sigmas), np.array(dummy_sigmas))

#Add date index
#from data_reader import read_all,apply_delta_t_to_data_frame,apply_z_score_to_data_frame,split_training_test
#df = read_all()
#df = apply_delta_t_to_data_frame(df)  # try to apply z-score before and after.
#dmean = np.mean(df)
#dstd = np.std(df)
#df = apply_z_score_to_data_frame(df, dmean, dstd)
#tr, te = split_training_test(df)  # we cheat a bit but very little, no problem.
#
#date_idx = te[10:].index

print('train_mape: '+'{:.2%}'.format(train_mape/100))
print('test_mape: '+'{:.2%}'.format(test_mape/100))
print('dummy_mape: '+'{:.2%}'.format(dummy_mape/100))


#trade vx
#Train Period: 04/12/2015-04/27/2015
#Trade Period: 04/24/2015-10/03/2017 predict value starts from 04/27/2015
#2 days ahead to apply delta t

vxx = read_vxx()
vxx = vxx.loc[vxx.index <= '2017-10-03 00:00:00']
vxx = vxx.loc[vxx.index >= '2015-04-24 00:00:00']

date_idx = vxx.index[1:]

#Plot Pictures
plt.figure(figsize=[12,8])
plt.plot(date_idx,true_sigmas,label='Observed')
plt.plot(date_idx,pred_sigmas,label='Predicted(LSTM)')
plt.legend(loc='upper right')
plt.title("Volatility Forecasting Made by the Long Short-Term Memory Model")
plt.show()

#Prepare trade data
prev_sigmas = [ z_score_inv(x_test[i][-1][0], mean, std) for i in range(len(x_test)) ]
vxx_df = pd.DataFrame(data = {'sigma': vxx})
#vxx_df = apply_delta_t_to_data_frame(vxx_df)
vxx_df = vxx_df.rename(index=str, columns={"sigma": "vxx"})
vxx_ret = pd.DataFrame(data = {"vxx_ret": np.log(vxx_df['vxx'].shift(-1)/vxx_df['vxx'])[1:]})

vxx_ret['vix'] = prev_sigmas
vxx_ret['pred_vix'] = pred_sigmas
vxx_ret['pred_ret'] = [ 1 if pred_sigmas[i] > prev_sigmas[i] else -1 for i in range(len(pred_sigmas))]

#dummy strategy
#vxx_ret['pred_ret'] = [ 1 if np.random.uniform(0,1) > 0.5 else -1 for i in range(len(pred_sigmas))]


vxx_ret['strategy_ret'] = vxx_ret['pred_ret']*vxx_ret['vxx_ret']
strategy_ret = vxx_ret['strategy_ret'].sum()


win_prob = vxx_ret[vxx_ret['strategy_ret']>0].shape[0]/vxx_ret.shape[0]

vxx_ret['true_vix'] = true_sigmas
vxx_ret['vix_ret'] = np.log(vxx_ret['true_vix']/vxx_ret['vix'])*vxx_ret['pred_ret']
vxx_ret[vxx_ret['vix_ret']>0].shape[0]/vxx_ret.shape[0]

pnl=vxx_ret['strategy_ret'].cumsum()
#plt.plot(pnl.index,pnl)
plt.figure(figsize=[12,8])
plt.plot(date_idx,pnl)
plt.show()

