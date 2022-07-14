#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 13:47:14 2022

@author: yajivluck
"""

import numpy as np


class Ball:
    
    dt = 0.1
    g = 9.81
    
    """
    Ball Class 
    """
    
    def __init__(radius = 1.0, color = 'y', x, y, vx, vy):
        
        self.x,self.y,self.vx,self.vy = x,y,vx,vy
        
    def new_position_velocity(x,y,vx,vy):
         
        new_x = x + vx*dt
        new_y = y + vy*dt - 0.5*g*dt**2
        new_vy = vy-g*dt
        return new_x,new_y,vx,new_vy
    
    
    
        
        
        
        
        
        
        
        
        
        
        