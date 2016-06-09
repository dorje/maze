# -----------------------------
# Author: Haoxi Zhang
# Version: 0.01 beta
# Date: 2016.06.09
# -----------------------------

import tensorflow as tf 
import numpy as np 
import random
from collections import deque 

# Hyper Parameters:

GAMMA = 0.99 # decay rate of past observations
OBSERVE = 1000. # timesteps to observe before training
EXPLORE = 50000. # frames over which to anneal epsilon
FINAL_EPSILON = 0 #0.001 # final value of epsilon
INITIAL_EPSILON = 1.0 #0.01 # starting value of epsilon
REPLAY_MEMORY = 5000 # number of previous transitions to remember
BATCH_SIZE = 64 # size of minibatch
UPDATE_TIME = 10

"""
I feel hard (particularly, in this code structure) 
to attach op like 'tf.scalar_summary("accuracy", self.accuracy)' 
to track 'accuracy' while training. 
So, I decided to wite it out in a external file:'temp.txt'.
And, if anyone knows how to use tensorboard op in this code for accuracy, 
please let me know. THANKS!
"""
#'temp.txt' is a file used to store the accuracy change during traning 
filename = "temp.txt"
f = open(filename,"w")

# Network Parameters

n_input = 8 # one-hot array of initial state : 8 total states
n_classes = 4 # four actions the agent can choose to act

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

class BrainDQN:

	def __init__(self, actions):
		# init replay memory
		self.replayMemory = deque()
		# init some parameters
		self.timeStep = 0
		self.epsilon = INITIAL_EPSILON
		self.actions = actions

		# init Q network
		self.stateInput, self.QValue, self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2  = self.createQNetwork()

		# init Target Q Network
		self.stateInputT,self.QValueT,self.W_fc1T,self.b_fc1T,self.W_fc2T,self.b_fc2T = self.createQNetwork()

		self.copyTargetQNetworkOperation = [self.W_fc1T.assign(self.W_fc1),self.b_fc1T.assign(self.b_fc1),self.W_fc2T.assign(self.W_fc2),self.b_fc2T.assign(self.b_fc2)]

		self.createTrainingMethod()

		# saving and loading networks
		self.saver = tf.train.Saver()
		self.session = tf.InteractiveSession()
		self.session.run(tf.initialize_all_variables())
		checkpoint = tf.train.get_checkpoint_state("saved_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.session, checkpoint.model_checkpoint_path)
			print "Successfully loaded:", checkpoint.model_checkpoint_path
		else:
			print "Could not find old network weights"
		
		
		# Add histogram summaries for weights
		tf.histogram_summary("w_h1_summ", self.W_fc1)
		tf.histogram_summary("w_h2_summ", self.W_fc2)

 		# Add histogram summaries for biases
		tf.histogram_summary("b_h1_summ", self.b_fc1)
		tf.histogram_summary("b_h2_summ", self.b_fc2)
        		
		#self.accuracy = self.getAccuracy()

		# Add scalar summary for cost
		tf.scalar_summary("cost", self.cost)
 
   		# create a log writer. run 'tensorboard --logdir=${PWD}'
    		self.writer = tf.train.SummaryWriter("./logs", self.session.graph) 
		self.merged = tf.merge_all_summaries()

	def createQNetwork(self):
        	# input layer
		stateInput = tf.placeholder("float",[None,8])

		W_fc1 = self.weight_variable([n_input,16])
		b_fc1 = self.bias_variable([16])

		W_fc2 = self.weight_variable([16, self.actions])
		b_fc2 = self.bias_variable([self.actions])

		# Q Value layer
		h_fc1 = tf.nn.relu(tf.matmul(stateInput,W_fc1) + b_fc1)
		QValue = tf.matmul(h_fc1,W_fc2) + b_fc2
		#QValue = tf.nn.relu(tf.matmul(stateInput, weights) + biases)

		return stateInput, QValue, W_fc1, b_fc1, W_fc2, b_fc2

	def copyTargetQNetwork(self):
		self.session.run(self.copyTargetQNetworkOperation)

	def createTrainingMethod(self):
		self.actionInput = tf.placeholder("float",[None,self.actions])
		self.yInput = tf.placeholder("float", [None]) 
		Q_Action = tf.reduce_sum(tf.mul(self.QValue, self.actionInput), reduction_indices = 1)
		self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
		self.trainStep = tf.train.AdamOptimizer(1e-2).minimize(self.cost)

	def trainQNetwork(self):	
		# Step 1: obtain random minibatch from replay memory
		minibatch = random.sample(self.replayMemory,BATCH_SIZE)
		state_batch = [data[0] for data in minibatch]
		action_batch = [data[1] for data in minibatch]
		reward_batch = [data[2] for data in minibatch]
		nextState_batch = [data[3] for data in minibatch]
 
		# Step 2: calculate y 
		y_batch = []
		QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT:nextState_batch})
	
		for i in range(0,BATCH_SIZE):
			terminal = minibatch[i][4]
			if terminal:
				y_batch.append(reward_batch[i])
			else:
				y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

		self.trainStep.run(feed_dict={
			self.yInput : y_batch,
			self.actionInput : action_batch,
			self.stateInput : state_batch
			})

		
		if self.timeStep % 20 == 0:
            		summary_str = self.session.run(self.merged,feed_dict={self.yInput : y_batch, self.actionInput : action_batch, self.stateInput : state_batch})
            		self.writer.add_summary(summary_str, self.timeStep)
 

		# save network every 100 iteration
		if self.timeStep % 10000 == 0:
			self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step = self.timeStep)

		if self.timeStep % UPDATE_TIME == 0:
			self.copyTargetQNetwork()
	
		if self.timeStep == 50001:
			print '500001'
			self.printExit()
		
	def setPerception(self,observation,action,reward,terminal):
		self.replayMemory.append((self.currentState,action,reward,observation,terminal))
		if len(self.replayMemory) > REPLAY_MEMORY:
			self.replayMemory.popleft()
		if self.timeStep > OBSERVE:
			# Train the network
			self.trainQNetwork()

		# print info
		state = ""
		if self.timeStep <= OBSERVE:
			state = "observe"
		elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
			state = "explore"
		else:
			state = "train"

		print "TIMESTEP", self.timeStep, "/ STATE", state, \
            	"/ EPSILON", self.epsilon

		self.currentState = observation
		self.timeStep += 1
		
		if self.timeStep % 20 == 0:
            		self.accuracy = self.getAccuracy()
			# Add scalar summary for accuracy
			#tf.scalar_summary("accuracy", self.accuracy)
			print 'Accuracy at step %s: %s' % (self.timeStep, self.accuracy) 
			f.write('Accuracy at step %s: %s \n' % (self.timeStep, self.accuracy))
 
		

	def getAction(self,actions):
		QValue = self.QValue.eval(feed_dict= {self.stateInput:[self.currentState]})[0]
		action = np.zeros(self.actions)
		action_index = 0
		if random.random() <= self.epsilon:
                	# x = len(actions)
			action_index = actions[ random.randrange(4) ]
			action[action_index] = 1
		else:
			action_index = np.argmax(QValue)
			action[action_index] = 1

		# change episilon
		if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
			self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE

		return action

	def setInitState(self,observation):
		self.currentState = observation

	def getAccuracy(self):
		accuracy = 0.0
		predict = [0,0,0,0,0,0,0]
		count = 0.0
		y = [3,0,3,3,1,3,0]
		for i in xrange(7):
			sInput = [0,0,0,0,0,0,0,0]
			sInput[i] = 1
			QValue = self.QValue.eval(feed_dict= {self.stateInput:[sInput]})[0]
		 	predict[i] = np.argmax(QValue)
		print predict
		for i in xrange(7):
    			if y[i] == predict[i]:
        			count +=1.0
		accuracy = count/7.0
		return accuracy

	def weight_variable(self,shape):
		initial = tf.truncated_normal(shape, stddev = 0.01)
		return tf.Variable(initial)

	def bias_variable(self,shape):
		initial = tf.constant(0.01, shape = shape)
		return tf.Variable(initial)

	def getPrintAction(self,a):
		action = 'none'
		if a == 0:
			action ='up'
		elif a == 1:
			action = 'down'
		elif a == 2:
			action = 'left'
		else:
			action = 'right'
		return action


	def printExit(self):
		print '\n Results: actions based on the QValue: '
		QValue = self.QValue.eval(feed_dict= {self.stateInput:[[1,0,0,0,0,0,0,0]]})[0]
		x = np.argmax(QValue)
		print 'in room 1, go: ' + self.getPrintAction(x)

		QValue = self.QValue.eval(feed_dict= {self.stateInput:[[0,1,0,0,0,0,0,0]]})[0]
		x = np.argmax(QValue)
		print 'in room 2, go: ' + self.getPrintAction(x)

		QValue = self.QValue.eval(feed_dict= {self.stateInput:[[0,0,1,0,0,0,0,0]]})[0]
		x = np.argmax(QValue)
		print 'in room 3, go: ' + self.getPrintAction(x)

		QValue = self.QValue.eval(feed_dict= {self.stateInput:[[0,0,0,1,0,0,0,0]]})[0]
		x = np.argmax(QValue)
		print 'in room 4, go: ' + self.getPrintAction(x)

		QValue = self.QValue.eval(feed_dict= {self.stateInput:[[0,0,0,0,1,0,0,0]]})[0]
		x = np.argmax(QValue)
		print 'in room 5, go: ' + self.getPrintAction(x)

		QValue = self.QValue.eval(feed_dict= {self.stateInput:[[0,0,0,0,0,1,0,0]]})[0]
		x = np.argmax(QValue)
		print 'in room 6, go: ' + self.getPrintAction(x)

		QValue = self.QValue.eval(feed_dict= {self.stateInput:[[0,0,0,0,0,0,1,0]]})[0]
		x = np.argmax(QValue)
		print 'in room 7, go: ' + self.getPrintAction(x)

		f.close()
		exit()

