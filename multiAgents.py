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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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

        result = 0
        if action == "Stop":
            return -float('inf')

        for elem in newScaredTimes:
            result += elem

        gMin = float('inf')

        for elem in newGhostStates:
            temp = manhattanDistance(elem.getPosition(), newPos)
            if (temp < gMin):
            	gMin = temp


        fMin = float('inf')
      
        for elem in newFood.asList():
            temp = manhattanDistance(newPos, elem)
            if (temp < fMin):
                fMin = temp

        return successorGameState.getScore() + result + (gMin / fMin)
       

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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        def max_value(state, currDepth):
            if state.isLose() or state.isWin() or currDepth == self.depth:
                return self.evaluationFunction(state)
            v = -float("inf")
            for elem in state.getLegalActions(0):
                v = max(v, min_value(state.generateSuccessor(0, elem), currDepth, 1))
            return v

        def min_value(state, currDepth, gIndex):
            if state.isLose() or state.isWin():
                return self.evaluationFunction(state)
            v = float("inf")
            for elem in state.getLegalActions(gIndex):
                newState = state.generateSuccessor(gIndex, elem)
                if (gIndex + 1) < gameState.getNumAgents():
                    v = min(v, min_value(newState, currDepth, gIndex + 1))
                else:
                	v = min(v, max_value(newState, currDepth + 1))
            return v

       
        firstTime = 1

        for elem in gameState.getLegalActions(0):
            newVal = min_value(gameState.generateSuccessor(0, elem), 0 , 1)
            if ((firstTime == 1) or (newVal > maximum)):
                maximum = newVal
                result = elem
                firstTime = 0
        return result
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def max_valueAB(state, currDepth, alpha, beta):
            if state.isLose() or state.isWin() or currDepth == self.depth:
                alpha = float('inf')
                return self.evaluationFunction(state)
            v = -float("inf")
            if (len(state.getLegalActions(0)) == 0):
            	return self.evaluationFunction(state)
            for elem in state.getLegalActions(0):

                v = max(v, min_valueAB(state.generateSuccessor(0, elem), currDepth, 1, alpha, beta))
                
                if (v > beta):
                    return v
                if (v > alpha):
                    alpha = v
            return v

        def min_valueAB(state, currDepth, gIndex, alpha, beta):
            if state.isLose() or state.isWin():
                return self.evaluationFunction(state)
            
            v = float("inf")

            for elem in state.getLegalActions(gIndex):
                if (gIndex + 1) < gameState.getNumAgents():
                	v = min(v, min_valueAB(state.generateSuccessor(gIndex, elem), currDepth, gIndex + 1, alpha, beta))
                else:
                    v = min(v, max_valueAB(state.generateSuccessor(gIndex, elem), currDepth + 1, alpha, beta))
                if (v < alpha):
                    return v
                if (v < beta):
                    beta = v
            return v

       
        firstTime = 1
        alpha = -float("inf")
        beta = float("inf")

        for elem in gameState.getLegalActions(0):
            newVal = min_valueAB(gameState.generateSuccessor(0, elem), 0 , 1, alpha, beta)
            if ((firstTime == 1) or (newVal > maximum)):
                maximum = newVal
                alpha = newVal
                result = elem
                firstTime = 0
        return result

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
        def max_valueEM(state, currDepth, alpha, beta):
            if state.isLose() or state.isWin() or currDepth == self.depth:
                alpha = float('inf')
                return self.evaluationFunction(state)
            v = -float("inf")
            if (len(state.getLegalActions(0)) == 0):
            	return self.evaluationFunction(state)
            for elem in state.getLegalActions(0):

                v = max(v, expected_valueEM(state.generateSuccessor(0, elem), currDepth, 1, alpha, beta))
                
            return v

        def expected_valueEM(state, currDepth, gIndex, alpha, beta):
            if state.isLose() or state.isWin():
                return self.evaluationFunction(state)
            
            v = float("inf")
            result = 0
            for elem in state.getLegalActions(gIndex):
                if (gIndex + 1) < gameState.getNumAgents():
                	v = expected_valueEM(state.generateSuccessor(gIndex, elem), currDepth, gIndex + 1, alpha, beta)
                else:
                    v = max_valueEM(state.generateSuccessor(gIndex, elem), currDepth + 1, alpha, beta)
                result += v
            return result / len(state.getLegalActions(gIndex))

       
        firstTime = 1
        alpha = -float("inf")
        beta = float("inf")

        for elem in gameState.getLegalActions(0):
            newVal = expected_valueEM(gameState.generateSuccessor(0, elem), 0 , 1, alpha, beta)
            if ((firstTime == 1) or (newVal > maximum)):
                maximum = newVal
                alpha = newVal
                result = elem
                firstTime = 0
        return result

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    currPos = currentGameState.getPacmanPosition()
    currGS = currentGameState.getGhostStates()

    gMin = float('inf')

    for ghost in currGS:
        temp = -manhattanDistance(currPos, ghost.getPosition())
        if (temp < gMin):
        	gMin = temp
    
    result = currentGameState.getScore() + gMin - 30 * len(currentGameState.getCapsules())

    return result

# Abbreviation
better = betterEvaluationFunction
