# A neural network is an inefficient method for mapping such a simple process 
# with a small number of discrete inputs and outputs, but I wanted to test it before
# applying the neural network to more difficult problems. It plots the average reward per timestep
# (maximum possible is 1) against training iterations. Training stops automatically once perfect performance
# is achieved over 100 trial games. Close the plot to render the game. 

import gym
import gym.spaces
import gym.wrappers
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from collections import deque
from keras.layers import Activation, Flatten, Dense
from keras import applications
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras import optimizers

def create_action_bins():
	actionslist = []
	for a in range(2):
		for b in range(2):
			for c in range(5):
				actionslist.append((a,b,c))
	
	# ~ print(actionslist)
	# ~ print(len(actionslist))
	return actionslist

def find_action_bin(action, action_bins):
	for i in range(len(action_bins)):
		# ~ print('i = ', i, 'action = ', action, 'action_bins[i] = ', action_bins[i])
		if action == action_bins[i]:
			return(i)
	print('error - no bin assigned')	

def build_model(num_output_nodes):
	
	model = Sequential()
	
	model.add(Dense(128, input_shape = (1,), activation = 'relu'))
	model.add(Dense(64, activation = 'relu'))
	model.add(Dense(64, activation = 'relu'))
	model.add(Dense(num_output_nodes, activation = 'softmax')) 
	
	adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
	
	model.compile(loss = 'mse', optimizer = adam)
	
	return model

def train_model(memory, gamma = 0.9):
	
	subset = memory
	
	for state, action_bin, reward, state_new in subset:
		
		# ~ print(type(state))
		
		# ~ print('state shape = ', np.shape(state))
		
		target = reward + gamma * np.amax(model.predict([state_new]))
		
		targetfull = model.predict([state])
		
		# ~ print('targetful ', targetfull)
		
		targetfull[0][action_bin] = target
		
		
		
		# ~ print('targetful ', targetfull)
		
		model.fit([state], targetfull, epochs = 1, verbose = 0) 

def run_episodes(action_bins, eps = 1, r = True):
	
	done = False
	
	memory = deque()
	
	
	state = env.reset()

	totalreward = 0

	cnt = 0
	
	while not done:
		
		cnt += 1
		
		# ~ print(cnt, 'state = ', state)
		
		if r:
			env.render()
			
		if np.random.uniform() < eps:
			action = env.action_space.sample()
			action_bin = find_action_bin(action, action_bins)
		else:
			action_bin = np.argmax(model.predict([state]))		
			action = action_bins[action_bin]
			
		# ~ print('action bin = ', action_bin)
		# ~ print('action = ', action)
		
		observation, reward, done, _ = env.step(action)  
			 
		totalreward += reward
		
		state_new = observation 
		
		memory.append((state, action_bin, reward, state_new))
		
		# ~ print('mem =', memory)
		
		state = state_new
		
	return memory

def play_games(action_bins, iters = 10, eps = 0.0,  r = True):
	
	rewardarray = []
	
	for i in range(iters):
		done = False
		state = env.reset()
		totalreward = 0
		cnt = 0
		# ~ print('test episode ', i)
		while not done:
			
			cnt += 1
			
			if r:
				env.render()
			
			action_bin = np.argmax(model.predict([state]))		
			
			# ~ print('play actionbin = ', action_bin)
			
			action = action_bins[action_bin]
			
			# ~ print('action = ', action)
			
			# ~ print('play action = ', action)
			
			# ~ print('play action_bins = ', action_bins)
			
			observation, reward, done, _ = env.step(action)  
			
			totalreward += reward
			
			state_new = observation 
			
			state = state_new
		
		rewardarray.append(totalreward/cnt)
		
	return rewardarray

if __name__ == '__main__':

	env = gym.make('Copy-v0')
	
	action_bins = create_action_bins()

	# ~ print('len action bins = ', len(action_bins))

	model = build_model(len(action_bins))
	
	eps = 1
	
	eps_decay = 0.99995
	
	
	totarray = []
	cntarray = []
	epsarray = []
	
	totaliters = 200000
	test_interval = 1000
	
	# ~ print('numeps = ', numeps)
	
	cnt = 0
	complete = False
	
	
	while cnt < totaliters and not complete:
		
		eps = eps * eps_decay	
		
		memory = run_episodes(action_bins, r = False)
		
		train_model(memory, gamma = 0.9)
		
		cnt += 1
		
		if cnt % test_interval == 0:
			print('iteration ', cnt, 'eps = ', eps)
		
			rewardarray = play_games(action_bins, iters = 100, r = False)
			cntarray.append(cnt)
			epsarray.append(eps)
			
			totarray.append(np.average(rewardarray))
			if np.average(rewardarray) == 1:
				complete = True
				print('Stopping training - perfect performance over 100 trials')
		
	plt.plot(cntarray, totarray)
	plt.plot(cntarray, epsarray)
	plt.show()
	
	
	x = input('Any key to play. Enter x to quit')
	while x != 'x':
		
		play_games(action_bins, iters = 1, r = True)
		x = input('Any key to play again. Enter x to quit')
		
	
	
	
	
	
	
		
	
		
