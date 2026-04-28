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
from pacman import GameState
from multiAgents import MultiAgentSearchAgent


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Agente que utiliza o algoritmo Minimax para tomar decisões.

    O Pac-Man é o agente MAXIMIZADOR: busca a ação que leva ao
    estado de maior utilidade (maior pontuação).

    Os fantasmas são agentes MINIMIZADORES: escolhem ações que
    levam ao estado de menor utilidade para o Pac-Man (pior cenário).

    A busca alterna entre os agentes a cada nível da árvore,
    incrementando a profundidade após todos os agentes jogarem
    uma rodada completa.
    """

    def getAction(self, gameState: GameState):
        #Substitui a função de avaliação padrão pela versão melhorada,
        #que considera comida restante, distância à comida e perigo dos fantasmas.
        self.evaluationFunction = betterEvaluationFunction

        def maxValue(state, depth):
            """
            Nó MAX — turno do Pac-Man (agente 0).

            O Pac-Man escolhe a ação que MAXIMIZA a utilidade esperada,
            assumindo que os fantasmas jogarão de forma ótima contra ele.

            Retorna sempre um valor numérico (score).
            """
            #Condição de parada: jogo encerrado ou profundidade máxima atingida
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            bestVal = -float('inf')  #inicia com o pior valor possível para o maximizador

            for action in state.getLegalActions(0):  #agente 0 = Pac-Man
                successor = state.generateSuccessor(0, action)
                #Após o Pac-Man agir, é a vez do primeiro fantasma (agente 1)
                score = minValue(successor, depth, 1)
                if score > bestVal:
                    bestVal = score

            return bestVal

        def minValue(state, depth, agentIndex):
            """
            Nó MIN — turno de um fantasma (agente 1 em diante).

            O fantasma escolhe a ação que MINIMIZA a utilidade do Pac-Man,
            assumindo que o Pac-Man jogará de forma ótima em seguida.

            Quando todos os fantasmas jogaram, passa o turno de volta
            ao Pac-Man e incrementa a profundidade.

            Retorna sempre um valor numérico (score).
            """
            #Condição de parada: jogo encerrado (não verifica profundidade aqui,
            #pois é verificado no início do turno do Pac-Man)
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()
            bestVal = float('inf')  #inicia com o melhor valor possível para o minimizador

            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)

                if agentIndex == numAgents - 1:
                    #Último fantasma da rodada: próximo é o Pac-Man,
                    #e a profundidade aumenta (uma rodada completa se passou)
                    score = maxValue(successor, depth + 1)
                else:
                    #Ainda há fantasmas para jogar nesta rodada
                    score = minValue(successor, depth, agentIndex + 1)

                if score < bestVal:
                    bestVal = score

            return bestVal
        
        #Aqui escolhemos a melhor AÇÃO (não só o valor) para o Pac-Man.
        #Iteramos sobre todas as ações legais e simulamos o que os
        #fantasmas fariam em resposta, escolhendo a ação de maior score.

        legalMoves = gameState.getLegalActions(0)  
        if not legalMoves:                         
            return Directions.STOP                

        bestAction = legalMoves[0]                
        bestVal = -float('inf')

        for action in legalMoves:                
            successor = gameState.generateSuccessor(0, action)
            #O primeiro nível após a raiz é sempre um nó MIN (turno dos fantasmas)
            score = minValue(successor, 0, 1)
            if score > bestVal:
                bestVal = score
                bestAction = action

        return bestAction


def betterEvaluationFunction(currentGameState: GameState):
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [g.scaredTimer for g in ghostStates]

    #Fator 1: quantidade de comida restante (menos é melhor)
    foodCount = len(food)

    #Fator 2: distância até a comida mais próxima (menor é melhor)
    foodDistances = [manhattanDistance(pos, f) for f in food]
    minFoodDistance = min(foodDistances) if foodDistances else 0

    #Fator 3: avaliação de perigo/oportunidade com fantasmas
    ghostPenalty = 0
    for i, ghost in enumerate(ghostStates):
        dist = manhattanDistance(pos, ghost.getPosition())
        if scaredTimes[i] > 0:
            #Fantasma assustado: vale a pena perseguir (desconta a penalidade)
            ghostPenalty -= 50 / (dist + 1)
        else:
            #Fantasma normal e perto: perigo alto, penaliza fortemente
            if dist < 3:
                ghostPenalty += 200 / (dist + 1)

    score = currentGameState.getScore()
    score -= 10 * foodCount        #penaliza comida restante
    score -= 2 * minFoodDistance   #penaliza distância da comida mais próxima
    score -= ghostPenalty          #penaliza perigo, recompensa oportunidade

    return score

better = betterEvaluationFunction
