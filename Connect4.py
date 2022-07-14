#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 11:06:49 2022

@author: yajivluck

Connect 4 game programmed in python (small project)
"""



import random


# Variable that stores if game is still going
game_still_going = True
 
# Variable that stores the winner
winner = None


#INSTANTIATION OF BOARD WITH DIMENSIONS

ROWS = 6
COLUMNS = 7
BOARD = [["." for x in range(COLUMNS)] for y in range(ROWS)]



# Variable that stores who's turn it is
players = ["X","O"]
current_player = random.choice(players)




 