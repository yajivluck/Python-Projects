# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 15:33:12 2023

@author: yajiluck
"""


class Player:
    
    def __init__(self, symbol = "X"):
        self.symbol = symbol
        
    def choose_move(self): 
        print(self.symbol)
        self.move = input("Choose a Move: ")
        print('\n')


