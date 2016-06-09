# -----------------------------
# Author: Haoxi Zhang
# Version: 0.01 beta
# Date: 2016.06.09
# -----------------------------

import sys
from Maze import Maze
from BrainDQN_Latte import BrainDQN
import numpy as np


# add some white noise to the input data 
# maybe unnecessary
def wgn(snr):
	snr = 10**(snr/10.0)
	t = np.arange(0, 8) * 0.1
	x = np.sin(t)
	xpower = np.sum(x**2)/len(x)
	npower = xpower / snr
	return np.random.randn(len(x)) * np.sqrt(npower)
	
def playMaze():
        
	# Step 1: init BrainDQN
	actions = 4
	brain = BrainDQN(actions)
	# Step 2: init Maze Game   
    	maze = Maze()
	# Step 3: play game
	# Step 3.1: obtain init state

	observation = [1,0,0,0,0,0,0,0]
	observation += wgn(7)

	brain.setInitState(observation)
	tt = 0 # number of termials
	# Step 3.2: run the game
	while 1:
		stateIndex = np.argmax(observation)
        	actions = [0,1,2,3] # the index of available actions: UP DOWN LEFT RIGHT
		action = brain.getAction(actions)
		reward,terminal, observation = maze.takeAction(action)
		observation += wgn(7)
		brain.setPerception(observation,action,reward,terminal)
		if terminal:
			tt+=1
			if tt == 1000:
				brain.printExit()
			print 'Terminal'
			maze.newGame()
			oberservation = [1,0,0,0,0,0,0,0]
			observation += wgn(7)
			brain.setInitState(observation)
        
        
def main():
	playMaze()

if __name__ == '__main__':
	main()
