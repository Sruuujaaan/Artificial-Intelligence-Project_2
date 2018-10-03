# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        pacToFoodDistances = []
        pacToGhostDistances = []
        foodPositionInMaze = newFood.asList()
        deciderVar = 0       # Helps in Determining if this state can be visited by Pac.

        # Calculate the distance from every foodPositionInMaze to current state in which pacman is and store it in the list.
        for food in foodPositionInMaze:
            pacToFoodDistances.append(manhattanDistance(newPos, food))

        # Change deciderVar according to food( Closer the food to Pac. better the state)
        for i in pacToFoodDistances:
            if i < 5:
                deciderVar += 1
            elif i >= 5 and i <= 15:
                deciderVar += 0.2
            else:
                deciderVar += 0.15

        # Calculate the distance from every Ghost in maze to current state in which pacman is and store it in the list.
        for ghost in successorGameState.getGhostPositions():
            pacToGhostDistances.append(manhattanDistance(newPos, ghost))

        # Change deciderVar according to ghost( Closer the ghost to Pac. Worse the state)
        for ghost in successorGameState.getGhostPositions():
            if ghost == newPos:     # Ghost is present on the next position
                deciderVar = 2 - deciderVar

            elif manhattanDistance(ghost, newPos) <= 3.5:
                deciderVar = 1 - deciderVar

        return successorGameState.getScore() + deciderVar

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        
        def miniMaxSearch(gameState, agentIndex, depth):
            nextMove = list()

            if not gameState.getLegalActions(agentIndex):   # If the state is illegal terminate it
                return self.evaluationFunction(gameState), 0

            if depth == self.depth:   # if reached max depth, evaluate state
                return self.evaluationFunction(gameState), 0

            if agentIndex == gameState.getNumAgents() - 1:   # start new max layer with bigger depth
                depth += 1
                nextAgent = self.index  # nextAgent = pacman

            else:
                nextAgent = agentIndex + 1  # Selecting nextAgent as a ghost

            # For every successor find minimax value
            for move in gameState.getLegalActions(agentIndex):

                if not nextMove:
                    nextValue = miniMaxSearch(gameState.generateSuccessor(agentIndex, move), nextAgent, depth)
                    # Store minimax value and move in nextMove list
                    nextMove.append(nextValue[0])
                    nextMove.append(move)
                else:
                    # Check if miniMaxSearch value is better than the previous one
                    previousValue = nextMove[0]  # Keep previous Minimax value.
                    nextValue = miniMaxSearch(gameState.generateSuccessor(agentIndex, move), nextAgent, depth)

                    # MaxAgent is Pacman
                    if agentIndex == self.index:
                        if nextValue[0] > previousValue:
                            nextMove[0] = nextValue[0]
                            nextMove[1] = move

                    # MinAgent is Ghost
                    else:
                        if nextValue[0] < previousValue:
                            nextMove[0] = nextValue[0]
                            nextMove[1] = move
            return nextMove

        # Intiallly minMaxSearch is called with depth = 0 and Pacman always plays first self.index.
        return miniMaxSearch(gameState, self.index, 0)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphaBetaPruning(gameState, agentIndex, depth, x, y):
            nextMove = list()

            if not gameState.getLegalActions(agentIndex):  # If the state is illegal terminate it
                return self.evaluationFunction(gameState), 0

            if depth == self.depth:  # if reached max depth, evaluate state
                return self.evaluationFunction(gameState), 0

            if agentIndex == gameState.getNumAgents() - 1:  # start new max layer with bigger depth
                depth += 1
                # Make a move and make sure that the first move is made by Pacman
                nextAgent = self.index  # nextAgent = pacman

            else:
                nextAgent = agentIndex + 1  # Selecting nextAgent as a ghost

            # For every successor find minmax value
            for move in gameState.getLegalActions(agentIndex):
                if not nextMove:
                    nextValue = alphaBetaPruning(gameState.generateSuccessor(agentIndex, move), nextAgent, depth, x, y)
                    # Store minimax value and move in nextMove list
                    nextMove.append(nextValue[0])
                    nextMove.append(move)

                    # Fixing x,y for the first node
                    if agentIndex == self.index:
                        x = max(nextMove[0], x)
                    else:
                        y = min(nextMove[0], y)
                else:
                    # Check if minMax value is better than the previous one so that some nodes can be ignored (Pruned)
                    # There is no need to search next nodes as alphaBetaPruning Pruning is true
                    if nextMove[0] > y and agentIndex == self.index:
                        return nextMove

                    if nextMove[0] < x and agentIndex != self.index:
                        return nextMove

                    previousValue = nextMove[0] # Keep previous value
                    nextValue = alphaBetaPruning(gameState.generateSuccessor(agentIndex, move), nextAgent, depth, x, y)

                    # MaxAgent is Pacman
                    if agentIndex == self.index:
                        if nextValue[0] > previousValue:
                            nextMove[0] = nextValue[0]
                            nextMove[1] = move
                            x = max(nextMove[0], x)

                    # MinAgent is Pacman
                    else:
                        if nextValue[0] < previousValue:
                            nextMove[0] = nextValue[0]
                            nextMove[1] = move
                            y = min(nextMove[0], y)
            return nextMove

        # Call alphaBetaPruning with initial depth = 0 and x ,y values as some large number (as it is infinity infinity)
        # return alphaBetaPruning(gameState,self.index,0, a big negative number, a big number)[1]
        return alphaBetaPruning(gameState, self.index, 0, -100000000, 100000000)[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

