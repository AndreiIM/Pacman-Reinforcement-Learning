# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py
# learningAgents.py

from pacman import Directions
from game import Agent
import random
import game
import util

# QLearnAgent
#
class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.2, epsilon=0.05, gamma=0.8, numTraining = 10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        self.previousState = None
        self.previousAction = None
        self.episodesSoFar = 0
        self.Q = util.Counter()

    
    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value
        
    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts
    # getAction
    #
    # The main method required by the game. Called every time that
    # Pacman is expected to move
    def getAction(self, state):

        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        #print "Reward: ", self.getReward()
        # Now pick what action to take. For now a random choice among
        # the legal moves
        #Check to see what kind of state Pacman is in
        self.verifyStateType(state)
        legalActions = state.getLegalPacmanActions()

        #Get best actions given the curent state
        finalAction = self.decideAction(legalActions, state)

        #Register previous state of affairs
        self.previousState = state
        self.previousAction = finalAction

        return finalAction


    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):
        #Adjust final values
        self.adjustFinalReward(state)
        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.episodesSoFar % 100 == 0:
            print 'Number of training Games:', self.episodesSoFar
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)
#############################################################
#                                                           #
#     Main Q-Learning Algorithm                             #
#############################################################

    #Method used to return the Q value dictionary
    #Initialy set to 0
    def getQValue(self, state, action):
        return self.Q[(state, action)]

    # Update the Q-Value dictionary using the current
    # And previous state of affairs
    def updateQValue(self, state, action, nextState, reward):
        transitionValue = self.alpha * (reward + self.gamma * self.getMaximumValueActionFromState(nextState))
        self.Q[(state, action)] = (1 - self.alpha) * self.Q[(state, action)] + transitionValue

    #Calculate the curent value for reward
    #And update the Q value dictionary
    def recomputeQValue(self, state):
        currentReward = state.getScore() - self.previousState.getScore()
        self.updateQValue(self.previousState, self.previousAction, state, currentReward)

    #Return the most valuble action give a state
    #If no legal action are available,or state is terminal, return 0
    def getMaximumValueActionFromState(self, state):
        pacmanActions = state.getLegalPacmanActions()
        bestAction = float("-inf")
        if len(pacmanActions) == 0:
            return 0.0
        for action in pacmanActions:
            bestAction = self.getBestAction(action, bestAction, state)
        return bestAction

    #Compute the best legal actions to take in
    #Pacman's current state
    #If there is more than one, pick a random one
    #If no legal actions are found just return None
    def computeBestActionPolicy(self, state):
        actions = state.getLegalPacmanActions()
        bestAction = float("-inf")
        bestActions = []
        bestActions = self.getBestPolicy(actions, bestAction, bestActions, state)
        if len(bestActions) == 0:
            return None
        else:
            return random.choice(bestActions)

    # Update the Q-Value dictionary
    # As long as it is not a terminal state
    def verifyStateType(self, state):
        if not self.previousState is None:
            self.recomputeQValue(state)

    # Based on the epsilon value
    # Decide if Pacman should explore or
    # Make the best available move
    def decideAction(self, legalActions, state):
        if self.decideToExplore(self.epsilon):
            return random.choice(legalActions)
        else:
            return self.computeBestActionPolicy(state)

#############################################################
#                                                           #
#     Auxiliary Methods for Q-Learning                      #
#############################################################
    #Method use by Pacman when deciding to explore
    def decideToExplore(self,value):
        randomValue = random.random()
        return randomValue < value

    #Compare between two actions and return the best one
    def getBestAction(self, action, bestAction, state):
        qValue = self.getQValue(state, action)
        if qValue > bestAction:
            bestAction = qValue
        return bestAction

    #From a legal set of actions for Pacman
    #And non-terminal state,return the best ones
    def getBestPolicy(self, pacmanActions, bestAction, bestActions, state):
        for action in pacmanActions:
            qValue = self.getQValue(state, action)
            if qValue == bestAction:
                bestActions.append(action)
            elif qValue > bestAction:
                bestActions = [action]
                bestAction = qValue
        return bestActions

    #Method called at the end of the episode
    #For one final reward and Q-Value update
    def adjustFinalReward(self, state):
        finalReward = state.getScore() - self.previousState.getScore()
        self.updateQValue(self.previousState, self.previousAction, state, finalReward)
