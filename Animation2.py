#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:42:54 2022

@author: yajivluck
#"""
#import matplotlib.pyplot as plt
#import numpy as np
#import matplotlib.animation as animation
#
#
#g = 9.8
#
#try:
#    u = float(input('Enter initial velocity (m/s): '))
#    theta = float(input('Enter angle (deg): '))
#except ValueError:
#    print('Invalid input.')
#else:
#    theta = np.deg2rad(theta)
#    
#
#t_flight = 2*u*np.sin(theta)/g
#t = np.linspace(0, t_flight, 100)
#x = u*np.cos(theta)*t
#y = u*np.sin(theta)*t - 0.5*g*t**2
#
#fig, ax = plt.subplots()
#line, = ax.plot(x, y, color='k')
#
#xmin = x[0]
#ymin = y[0]
#xmax = max(x)
#ymax = max(y)
#xysmall = min(xmax,ymax)
#maxscale = max(xmax,ymax)
#circle = plt.Circle((xmin, ymin), radius=np.sqrt(xysmall))
#ax.add_patch(circle)
#
#def update(num, x, y, line, circle):
#    line.set_data(x[:num], y[:num])
#    circle.center = x[num],y[num]
#    line.axes.axis([0, max(np.append(x,y)), 0, max(np.append(x,y))])
#
#    return line,circle
#
#ani = animation.FuncAnimation(fig, update, len(x), fargs=[x, y, line, circle],
#                              interval=25, blit=True)
#
##ani.save('projectile.mp4')
#plt.show()


"""
Matplotlib Animation Example

author: Jake Vanderplas
email: vanderplas@astro.washington.edu
website: http://jakevdp.github.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
line, = ax.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    x = np.linspace(0, 2, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=20, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()