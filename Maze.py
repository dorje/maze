# -----------------------------
# Author: Haoxi Zhang
# Version: 0.01 beta
# Date: 2016.06.09
# -----------------------------

"""
The maze game
There are eight room in the maze, like below: 
 ___________________
|     |             |
|  3  |  6  |       |
|  _________|   8   |
|     |     |       |
|  2  |  5  |       |
|___  |  ___|____   |
|     |     |       |
|  1  |  4  |   7   |
|___________________|

and
the agent always starts from room 1,
the aim is to reach room 8

the agent can do four actions:
up, down, left, right
and there is no direction difference 
"""


import numpy as np

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
STATES = 8

class Maze:
    
    def __init__(self):
        
        self.agentState = 0
        self.moves = 0
        self.actionMap = [[UP, RIGHT], 
                          [UP, DOWN], 
                          [DOWN, RIGHT], 
                          [LEFT, RIGHT, UP],
                          [DOWN], 
                          [LEFT, RIGHT],
                          [LEFT, UP],
                          [LEFT,DOWN]]
        
        self.statMap = [[1,3],
                        [2,0],
                        [1,5],
                        [0,6,4],
                        [3],
                        [2,7],
                        [3,7],
                        [5,6]]
        
    def newGame(self):
        self.agentState = 0
        self.moves = 0
    
    def getAvailableActions(self, _s):
        return self.actionMap[_s]
    
    def takeAction(self, _a):
        self.moves +=1
        terminal = False
        action_ID = np.argmax(_a)
        nextState_ID = self.agentState
        nextState_OB = np.zeros(STATES)
        index_ns_ob = 0
        reward = 0
        try:
            index_value = self.getAvailableActions(self.agentState).index(action_ID)
        except ValueError:
            index_value = -7
        if index_value != -7:
            nextState_ID = self.statMap[self.agentState][index_value]
            self.agentState = nextState_ID
            if self.agentState == 7 :
                reward = 100
                terminal = True
        else:
            reward = 0#-1

        nextState_OB[nextState_ID] = 1
        return reward, terminal, nextState_OB
