# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 19:15:52 2023

@author: Kiran
"""
import pandas as pd
from tabulate import tabulate
import copy
import math
import random
import numpy as np



#Defining a Piece Class for Each Piece of the Dynamic Connect4 Game
class Piece:
    
    def __init__(self, x, y, symbol):
        
        self.x = x
        self.y = y
        self.symbol = symbol
        self.max_distance = 0
        

    #Move piece in board
    def move(self,direction, distance):
        #Modify coords of position based on direction/distance combination
        if direction == "N":
            self.y -=distance
        elif direction == "S":
            self.y+=distance
        elif direction == "W":
            self.x-=distance
        elif direction == "E":
            self.x+=distance
        else:
            print("Invalid Motion")
     

class BoardRule:
    
    def __init__(self, board = None, maximizing = True):
        # Define a 7x7 board that the game will be played on
        
        if maximizing == None:
            self.maximizing = True
        else:
            self.maximizing = maximizing
            
        self.Winner = None
        
        if board is None:
            
            
            # board = [["","X","X","","O","",""],
            #           ["","","","","","","X"],
            #           ["","","O","O","","",""],
            #           ["","","O","","","X","O"],
            #           ["","","","","","",""],
            #           ["","X","","","","",""],
            #           ["","","","O","X","X",""]
            #         ]
            
            
            
            
            board = [["","X","X","","O","",""],
                      ["","","","","","","X"],
                      ["O","","","","","",""],
                      ["O","","","","","","O"],
                      ["","","","","","","O"],
                      ["X","","","","","",""],
                      ["","","O","","X","X",""]
                    ]
            
            # board = [["","O","","","X","X","X"],
            #           ["","","","","X","O","X"],
            #           ["","","","","","",""],
            #           ["O","","","","","",""],
            #           ["","","","","","",""],
            #           ["","","","","","","O"],
            #           ["O","O","","","","","X"]
            #         ]
            
            
            # board = [["","","","","","","O"],
            #           ["","","","","","","X"],
            #           ["","","","","","X","X"],
            #           ["","","","","","","O"],
            #           ["","","","","","","X"],
            #           ["","O","","","","","O"],
            #           ["X","X","O","O","","",""]
            #         ]
            
            # board = [["","","","","","O","O"],
            #           ["","","","","O","X","X"],
            #           ["","","","","","",""],
            #           ["","","","","","O",""],
            #           ["","","","","","",""],
            #           ["","X","","","","X","O"],
            #           ["","X","X","","","","O"]
            #         ]
        
        self.board = pd.DataFrame(board)
        #Build List of pieces object that board holds
        self.pieces = [Piece(y + 1, x + 1, self.board.loc[x, y]) for x in self.board.index for y in self.board.columns if self.board.loc[x, y] != ""]
        #Set max distance for each pieces, score, and winner is there is one
        self.update_board()
        
    #Overwrite print function of Board Object to print board in aesthetically pleasing way
    def __str__(self):
        
        board = pd.DataFrame('', index=range(7), columns=range(7))
        for piece in self.pieces:
            board[piece.x - 1][piece.y - 1] = piece.symbol
            
        self.board = board
        
        return "  1,  2,  3,  4,  5,  6,  7" + '\n' + tabulate(self.board, tablefmt="grid", showindex=False) + '\n'
    
    
    def make_copy(self):
      # Create a deep copy of the current board instance
      board_copy = copy.deepcopy(self)

      return board_copy
        
        
    #This evaluates new max_distance for each piece on the board
    def update_max_distance(self):
        #Coords/Symbol of current piece
        #For each pieces in the borad
        for cur_piece in self.pieces:
            opponent_count = 0
            
            for other_piece in self.pieces:
                #Ignore self and same team Pieces
                if cur_piece == other_piece or cur_piece.symbol == other_piece.symbol: continue
                
                #opponent coordinates
                opp_x,opp_y = other_piece.x, other_piece.y
                #Piece is too far on x axis
                if opp_x > cur_piece.x + 1 or opp_x < cur_piece.x -1: continue
                #Piece is too far on y axis
                if opp_y > cur_piece.y + 1 or opp_y < cur_piece.y -1: continue
                #If Opp Piece pass all continue gates, it is adjacent to current piece
                #Add opponent_count as piece is adjacent
                opponent_count+=1
                
            #Define current piece max_distance
            if opponent_count >= 3:
                cur_piece.max_distance = 0      
            elif opponent_count == 2:
                cur_piece.max_distance = 1
            elif opponent_count == 1:
                cur_piece.max_distance = 2
            else: cur_piece.max_distance = 3
            
    #Make move on board copy of current state and returned board's copy with moved state 
    def make_move(self,move):
        #Make copy of current board
        
        #Expand Move String
        x_start,y_start,direction,distance = move[:4]
        x_start = int(x_start)
        y_start = int(y_start)
        distance = int(distance)
        cur_piece = None
        
        for piece in self.pieces:
            #Iterate through board piece until piece that corresponds to move is found
            if piece.x == x_start and piece.y == y_start:
                cur_piece = piece
                break
            
        if cur_piece is not None:
            #Move piece in direction at distance
            cur_piece.move(direction,distance)
            #Update Board After Move
            self.maximizing = not self.maximizing
            self.update_board()
                
        else:
            print("Invalid Move")
        
    #Move to update board metrics
    def update_board(self):
        #Default maximizing player as starting player else alternade players
        self.update_max_distance()
        #Check if a winner occured after the last move
        self.Winner = self.is_terminal()
        #Update score value of board after verifying/updating if a win occurred
        self.score = self.evaluate_board()
        
        
        
    #Maximize score for squares being built and minimize score for opponent square being built. As square is being build.
    #Compound score with potential pieces to complete the square. Penalize pieces that aren't potential square completions
    
    #Scale score of potential by distance (x + y) of completed square's coords
    #Scale down score of piece if opponent pieces are in the way to get that completed square
        
    #TODO FOR FUN
    
    
    def evaluate_board_final(self):
        
        if self.Winner == "O":
            #print('x wins state')
            return math.inf
        elif self.Winner == "X":
            #print('o wins state')
            return -math.inf
        
        board = np.zeros((7, 7), dtype=str)
        #Build numpy 2d array for current board state
        for piece in self.pieces:
            x, y, symbol = piece.x, piece.y, piece.symbol
            board[y - 1, x - 1] = symbol
            
            
        for x,y in board[x,y]:
            pass
            
           
    #Simplistic evaluate_board function
    # def evaluate_board(self):
        
    #     if self.Winner == "O":
    #         return 1
    #     elif self.Winner == "X":
    #         return -1
    #     else: return 0
        
            
            
        
        

    #Function evaluates score of current board state. By Convention I am making the starting player "X" as the
    #Maximimizing player and "O" as the minimizing player
    
    def evaluate_board(self):
        #If there is a winner on the board's state, return appropriate value
        if self.Winner == "O":
            #print('x wins state')
            return math.inf
        elif self.Winner == "X":
            #print('o wins state')
            return -math.inf
        
        else:
            
            symbol_values = {"O": 1, "X": -1}  # You can extend this dictionary as needed

            
            #Sum Distance between max and min player (higher score means mobile "O" lower score means mobile "X")
            #TODO Make better evaluating function
            #distance_score = 0
            #distance_scale = -0.5
            #Score for same element pieces being closer together (should have highest scale)
            proximity_score = 0
            proximity_scale = 1.2
            #Score for pieces more centered
            center_score = 0
            center_scale = 2.1
            #Score for Pieces around the Edges (Penalize edges?)
            edge_score = 0
            edge_scale = -1.5
            
            
            
           
            
            for piece in self.pieces:
                #for other_piece in self.pieces:
                    
    
                #Adds or Substract based on piece symbol
                scale = symbol_values.get(piece.symbol)
                #Priorise "freer" pieces (add score for "X", substract for "O" convention for all metrics here)
                
                #distance_score += scale * piece.max_distance * distance_scale
                #If piece within square   
                if 3 <= piece.x <= 5 and 3 <= piece.y <= 5:
                    
                    if piece.x == piece.y == 4:
                        center_score += scale * 2*center_scale 
                        
                    else:
                        center_score += scale * center_scale
                #If Edge Piece
                elif ((piece.x == 1 or piece.x == 7) and (piece.y == 1 or piece.y == 7)):
                    edge_score += scale * edge_scale
                    
                #Symbol of current piece 3x3 being evaluated
                symbol = piece.symbol
                
                ally_count = 0
                opponent_count = 0
                
                for other_piece in self.pieces:
                    #Ignore same piece
                    if other_piece == piece: continue
                    
                
                    #If other piece within 3x3 square
                    if (piece.x - 2 <= other_piece.x <= piece.x + 2) and (piece.y - 2 <= other_piece.y <= piece.y + 2):
                        
                        #Add ally/opponent by 0.2 for pieces in 4x4 vicinity
                        if other_piece.symbol == symbol: ally_count +=0.2
                        else: opponent_count +=0.2
                        
                    #3x3 Vicinity
                    elif (piece.x - 1 <= other_piece.x <= piece.x + 1) and (piece.y - 1 <= other_piece.y <= piece.y + 1):
                        #Add ally/opponent by 0.2 for pieces in 4x4 vicinity
                        if other_piece.symbol == symbol: ally_count +=0.8
                        else: opponent_count +=0.8
                        
                        
                        
                    elif (piece.x == other_piece.x and (other_piece.y + 1 == piece.y or piece.y == other_piece.y -1)) or \
                        (piece.y == other_piece.y and (other_piece.x + 1 == piece.x or piece.x == other_piece.x - 1)):
                            if other_piece.symbol == symbol: ally_count +=2
                            else: opponent_count+=2
                        
                    #Punish pieces too far
                    else:
                        ally_count = 0
                        opponent_count = 3
                        
                        
                        
                        
                        
                        
              
                        
                        
                    proximity_score += scale * proximity_scale * (ally_count - opponent_count)
                        
                        
                        
               

                    
               
                        
                
                
                
                

                #(3,3) - (5,3)
                #(3,5) - (5,5)
  
            #return distance_score + center_score + edge_score + proximity_score
            return center_score + edge_score + proximity_score

        
        
        
    def distance(self,piece_1,piece_2):
        
        x1,y1,x2,y2 = piece_1.x,piece_1.y,piece_2.x,piece_2.y
        return max(x1,x2) - min(x1,x2) + max(y1,y2) - min(y1,y2)
        

    #Search for 2x2 square assuming piece is top left of square (if a square exists, then finding it by guessing a piece is the top left piece of
    #the square will return a square if it exists)
    #This function works assuming UNIQUE set of coordinates per symbol (no two pieces in board can have the same coordinates)
    def is_terminal(self):
        
        for top_left_piece in self.pieces:
            #Assume no adjacent pieces to current potential top_left at start
            adjacent_count = 0
            
            for other_piece in self.pieces:
                #Ignore same piece when looking for other pieces to complete square
                if other_piece == top_left_piece: continue

                #Conditions to check if piece that could make up square prevents square from being made
                #If none of these conditions are true, other_piece is not potential square piece
                
                
                #Ignore pieces to its left or above as we assume top_left_piece as top left of current square
                if other_piece.x < top_left_piece.x or other_piece.y < top_left_piece.y: continue
                #Other piece coordinates too far from potential top_left to be considered
                if other_piece.x > top_left_piece.x + 1 or other_piece.y > top_left_piece.y + 1: continue
                #If adjacent but not same team piece, impossible square, break
                if other_piece.symbol != top_left_piece.symbol: break
                
                #Adjacent piece of same symbol, add one to adjacent count 
                #(if it reaches three assuming no two pieces can have the same coordinates on a board, the three
                #Adjacent pieces will form a square, return the square's symbol to determine winner
                else: adjacent_count+=1
                if adjacent_count == 3:
                    return top_left_piece.symbol
                
            #print(top_left_piece.x,top_left_piece.y,top_left_piece.symbol,adjacent_count)

        #If iterating over all potential top left piece yields no square, return None (no square found)
        return None
        
        
    #Returns list of possible moves based on what player's turn it is
    def get_possible_moves(self):
        # Implement this function to return possible moves
        possible_moves = []
        #If there is a winner on the current board state, no possible moves
        if self.Winner:
            return possible_moves
        
        if self.maximizing:
            symbol = "O"
        else: symbol = "X"
   
        #For all piece
        for piece in self.pieces:
            #Only look at own pieces
            if piece.symbol != symbol: continue
            #For each directions
            for direction in ["N","S","W","E"]:
                #For distances in piece max_distance
                for distance in range(1,piece.max_distance + 1):
                    #Build potential legal move based on format accepted
                    potential_move = str(piece.x) + str(piece.y) + direction + str(distance)
                    #If valid move, add to potential moves
                    if self.isvalidmove(potential_move):
                        possible_moves.append(potential_move)
                        
                        
        random.shuffle(possible_moves)
        
        return possible_moves  
        
    
    #Boolean Function that checks whether a move is valid
    def isvalidmove(self, move, show_validity = False):
        
        symbol = "O" if self.maximizing else "X"
        
        try:
            assert len(move) == 4
            x_start,y_start,direction,distance = move
            assert direction in ["N","S","W","E"]
            x_start, y_start, distance = int(x_start), int(y_start), int(distance)
            assert 0 < distance <= 3
            cur_piece = None
            
            if show_validity: print('valid format')
            
            for piece in self.pieces:
                #Ignore opponent's pieces
                if piece.symbol != symbol: continue
            
                if piece.x == x_start and piece.y == y_start:
                    if distance <= piece.max_distance:
                        cur_piece = piece
                        break
                    else: raise AssertionError
                
                   
            #Make sure a piece exists at initial location
            assert cur_piece is not None
            #Valid Start Position for a move
            
            if show_validity: print('your piece exists')
            
            new_x = x_start
            new_y = y_start
        
            #Direction already asserted as part of directions, make sure within bounds for new end coords
        
            if direction == "N":
                new_y-=distance
                if new_y < 1: raise AssertionError
                
            elif direction == "S":
                new_y+=distance
                if new_y > 7: raise AssertionError

            elif direction == "W":
                new_x-=distance
                if new_x < 1: raise AssertionError

            else:
                new_x+=distance
                if new_x > 7: raise AssertionError
                
            if show_validity: print('inbound motion')

            #Based on potential new coordinate, assert if valid motion that doesn't cross other pieces
            assert self.piece_cross(cur_piece, new_x, new_y, show_validity = show_validity)
            
            if show_validity: print('no collision')
            
            #Valid start piece pos for player, valid distance, valid direction, final pos in bounds and no overlaps
            
            
            return True
                
        
        except AssertionError:
            if show_validity: print("Invalid move: ", move)
            return False

    #Given pair of coordinate, Piece and Board, function returns bool of motion crossing
    #Other pieces
    def piece_cross(self, cur_piece : Piece, x2, y2, show_validity = False):
        #Coordinate of piece being evaluated for crossing
        x1,y1 = cur_piece.x, cur_piece.y
        
        #Left/Right Motion
        if x1 != x2 and y1 == y1:
            coordinate_motion = [(x, y1) for x in range(min(x1, x2), max(x1, x2) + 1)]
        
        else: coordinate_motion = [(x1, y) for y in range(min(y1, y2), max(y1, y2) + 1)]
   
   
        if show_validity: print(coordinate_motion)
   
        for piece in self.pieces:
            #Ignore self when comparing to other pieces
            if piece == cur_piece: continue
        
            #coordinate of piece being checked
            x_piece,y_piece = piece.x,piece.y
        
            #For all coordinates that piece is moved through
            for x,y in coordinate_motion:
                #If piece in way of motion (ignoring starting position)
                
                #Invalid Motion as collision happens
                if x_piece == x and y_piece == y: return False
                
        #Valid Motion
        return True





