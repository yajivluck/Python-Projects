# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 15:33:12 2023

@author: yajiluck
"""

from MinimaxAlphaBeta import BoardGame


from BoardRule import BoardRule
#Convention is "X" is maximizing player (first to play)
#"O" is minimizing player (second to play)
class Agent:
    
    def __init__(self, board_state : BoardRule, symbol, depth):
        
        
        self.board_state = board_state
        #Make minimax object with current board being played
        self.minimax = BoardGame(board = self.board_state)
        self.symbol = symbol
        self.depth = depth
        self.move = None
        
        
    #TODO IMPLEMENT TIME CONSTRAINT IN MOVE CHOOSING TO LIMIT TO 7-8 Seconds (make sure move sends before 10 seconds)
        
    def choose_move(self):       
        if self.symbol == "O":
            self.move =  self.minimax.find_best_move_maximizing(max_depth = self.depth)

        else:
            self.move = self.minimax.find_best_move_minimizing(max_depth = self.depth)        

        return self.move 
        

    

#Board = BoardRule()

#Minimax = BoardGame(Board)

#best_max = Minimax.find_best_move_maximizing()
# best_min = Minimax.find_best_move_minimizing()
    
        
    