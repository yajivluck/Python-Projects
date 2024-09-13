# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 17:30:47 2023

@author: yajiluck
"""
import math
from BoardRule import BoardRule
import numpy as np


#state_count = 0

class BoardGame:
    def __init__(self, board : BoardRule):
        # Initialize your game board, players, and other necessary variables here.
        self.board = board
     
        
        
    def evaluate_board(self):
        # Define your evaluation function here.
        # This function `should return a score that represents the state of the game.
        # Positive values indicate an advantage for the maximizing player,
        # while negative values indicate an advantage for the minimizing player.
        
        return self.board.evaluate_board()
        

    def is_terminal(self):
        # Define a terminal condition for your game.
        # Return True if the game is over, False otherwise.
        
        Winner = self.board.is_terminal() 
        return True if Winner else False
        

    def get_possible_moves(self):
        # Return a list of all possible moves for the given player on the current board.
        #If maximizing player's turn, return X possible Moves, else O possible Moves        
        
        return self.board.get_possible_moves()
    
    def make_move(self, move):
        # Apply the move to the board and return the new board state.
        self.board.make_move(move)

    def maximize(self, alpha, beta, depth, board):
        
        #global state_count 
        #state_count+=1
        
        if board.is_terminal() or depth == 0:
            return board.evaluate_board(), depth

        max_eval = -math.inf
        min_depth = -math.inf
        
        for move in board.get_possible_moves():
            next_state = board.make_copy()
            next_state.make_move(move)
            eval, new_depth = self.minimize(alpha, beta, depth - 1, next_state)
            
            #print(eval,move,depth)
            
            if eval >= max_eval:
                
                #Faster win same eval
                if eval == max_eval and new_depth > min_depth:
                    max_eval = eval
                    min_depth = new_depth
                    alpha = max(alpha, eval)
                #Slower win same eval 
                elif eval == max_eval and depth <= min_depth:
                    continue
                #More winning (not win guaranteed yet)
                else:
                    max_eval = eval
                    min_depth = depth
                    alpha = max(alpha, eval)

            

            #TODO REMOVE WHEN TESTING

            if beta <= alpha:
                break

        return max_eval, min_depth

    def minimize(self, alpha, beta, depth, board):
        
        #global state_count 
        #state_count+=1
        
        if board.is_terminal() or depth == 0:
            return board.evaluate_board(), depth

        min_eval = math.inf
        min_depth = -math.inf
        for move in board.get_possible_moves():
            next_state = board.make_copy()
            next_state.make_move(move)
                        
            eval, new_depth = self.maximize(alpha, beta, depth - 1, next_state)
            
            #print(eval,move,depth)

            
            if eval <= min_eval:
                
                if eval == min_eval and new_depth > min_depth:
                    min_eval = eval
                    min_depth = new_depth
                    beta = min(beta,eval)
                    
                elif eval == min_eval and depth <= min_depth:
                    continue
                
                else:
                    min_eval = eval
                    min_depth = new_depth
                    beta = min(beta,eval)
                    
                    
            #TODO REMOVE WHEN TESTING
            if beta <= alpha:
                break

        return min_eval, min_depth

    def minimax(self, depth, board):
        #Get Maximizing state of board from board
        maximizing_player = board.maximizing
        
        if maximizing_player:
            return self.maximize(-math.inf, math.inf, depth, board)
        else:
            return self.minimize(-math.inf, math.inf, depth, board)
        
        
    def find_best_move_minimizing(self, max_depth = 1):
        
        #global state_count
        best_move = None
        best_score = math.inf
        fastest_win = -math.inf
        #TODO SLOWEST LOSS DEPTH CLOSEST TO 0
        slowest_loss = math.inf
        #Assume Board state is minimizing player's turn
        self.board.maximizing = False
        
        for move in self.get_possible_moves():
            
            next_state = self.board.make_copy()
            next_state.make_move(move)
            
            
            #Assume next player's turn is maximizing player
            score,depth = self.minimax(max_depth - 1, board = next_state)
            
            #print(move,score,depth)
            
            
            #delay loss
            if score == math.inf and slowest_loss > depth:
                slowest_move = move
                slowest_loss = depth
                
                        
            elif score <= best_score:
                #Faster Win
                if score == best_score and depth > fastest_win:
                    fastest_win = depth 
                    best_score = score
                    best_move = move
                #Slower/Equal Speed Win
                elif score == best_score and depth <= fastest_win:
                    continue
                #Better Win
                else:
                    best_score = score
                    best_move = move
                    fastest_win = depth 



    
        #print('visited', state_count, 'states')
        #state_count = 0
        
        #If inevitable loss
        if best_score == math.inf: return slowest_move

        if best_move == None: return move
        

        return best_move
    

    def find_best_move_maximizing(self, max_depth = 1):
        
        #global state_count
        
        best_move = None
        best_score = -math.inf
        fastest_win = -math.inf
        #TODO SLOWEST LOSS, depth closest to 0 if guaranteed loss
        slowest_loss = math.inf

        #Assume board's state is maximizing player's turn
        self.board.maximizing = True

        for move in self.get_possible_moves():
            
            
            
            next_state = self.board.make_copy()
            next_state.make_move(move)
            
            #Assume next player's turn is minimizing player
            score,depth = self.minimax(max_depth - 1, board = next_state)
            
            #print(move,score,depth)
            
            # print(move,score)
            
            #delay loss
            if score == -math.inf and slowest_loss > depth:
                slowest_move = move
                slowest_loss = depth
                
        
            if score >= best_score:
                
                #Faster win
                if score == best_score and depth > fastest_win:
                    fastest_win = depth
                    best_score = score
                    best_move = move
                
                #Slower/Equal speed win
                elif score == best_score and depth <= fastest_win:
                    continue
                    
                #Better Win
                else:         
                    fastest_win = depth
                    best_score = score
                    best_move = move
                    
                    
        #print('visited', state_count, 'states')
        #state_count = 0

        if best_score == -math.inf: return slowest_move
        
        if best_move == None: return move

        return best_move
    
    
    
    # def find_best_move_minimizing(self, max_depth = 1):
    #     best_move = None
    #     best_score = math.inf
    #     #Assume Board state is minimizing player's turn
    #     self.board.maximizing = False
        
    #     for move in self.get_possible_moves():
            
    #         next_state = self.board.make_copy()
    #         next_state.make_move(move)
            
            
            
    #         #Assume next player's turn is maximizing player
    #         score = self.minimax(max_depth - 1, board = next_state)
            
    #         #print(move,next_state,score)
            
    #         if score <= best_score:
    #             best_score = score
    #             best_move = move
    #     return best_move
    

    # def find_best_move_maximizing(self, max_depth = 1):
    #     best_move = None
    #     best_score = -math.inf
    #     #Assume board's state is maximizing player's turn
    #     self.board.maximizing = True

    #     for move in self.get_possible_moves():
            
    #         next_state = self.board.make_copy()
    #         next_state.make_move(move)
            
    #         #Assume next player's turn is minimizing player
    #         score = self.minimax(max_depth - 1, board = next_state)
            
    #         # print(move,score)

        
    #         if score >= best_score:
    #             best_score = score
    #             best_move = move

    #     return best_move

