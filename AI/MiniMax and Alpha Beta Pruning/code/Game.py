# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:41:59 2023

@author: Kiran
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 14:07:56 2023

@author: yajiluck
"""
from Player import Player 
from BoardRule import BoardRule
from Agent import Agent



class Game:

    def __init__(self, Player_1 = "Human", Player_2 = "Agent", depth = 3):
        
        
        self.Board = BoardRule()
        
        if Player_1 == "Human" :
            self.Player_1 = Player(symbol = "O")
            
        elif Player_1 == "Agent":
            self.Player_1 = Agent(board_state = self.Board, symbol = "O", depth = depth)
            
        else:
            print("Invalid Player1, select Human or Agent")
            return
        
        
        if Player_2 == "Human":
            self.Player_2 = Player(symbol = "X")
            
        elif Player_2 == "Agent":
            self.Player_2 = Agent(board_state = self.Board, symbol = "X", depth = depth)
            
        else:
            print("Invalid Player2, select Human or Agent")
            return
    
    def start(self):
      
        #Print board at start
        print(self.Board)


        try:
            #Play as long as there is no winner
            while not self.Board.Winner:
                
                # Inserting Their next move on the board
                  
                while True:
                    self.Player_1.choose_move()
                    if self.Board.isvalidmove(self.Player_1.move):
                        self.Board.make_move(move = self.Player_1.move)
                        print("Player1 Plays:",self.Player_1.move,'\n')
                        print(self.Board,self.Board.score)
                        break
                   
                
                Winner = self.Board.is_terminal()
                if Winner is not None:
                    break              
                
                # Inserting Their next move on the board
                #print(self.Board)
            
                while True:
                    self.Player_2.choose_move()
                                        
                    if self.Board.isvalidmove(self.Player_2.move):
                        self.Board.make_move(move = self.Player_2.move)
                        print("Player2 Plays:",self.Player_2.move)
                        print(self.Board,self.Board.score)

                        break
                    
                    
                
                
             
                Winner = self.Board.is_terminal()
                if Winner is not None:
                    break  
                
                # Inserting Their next move on the board
                

            print(self.Board)
            print(Winner, 'wins!')
            print(self.Board.score)
    
        except:
            KeyboardInterrupt()
            
            
#Game = Game(Player_1= "Human", Player_2 = "Agent")
Game = Game(Player_1= "Agent", Player_2 = "Agent")

Game.start()






