import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

#functions = [np.sin(t),np.cos(t),np.log(t),np.exp(t),np.tan(t),np.arccosh(t)]

CURRENT = int(time.time())
FRAMES = 100 #Number of frame per functions
INTERVALS = 0.1  #Seconds between each frames
TIME_CUT = 2 #Time between switch of functions

#Generate initial dimensions of animation window
def init():
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(0, 10)
    del xdata[:]
    del ydata[:]
    line.set_data(xdata, ydata)
    return line,

#Generate data from a function (Only care about the value of the function at the current time, past values don't need to be stored in memory -> yield)
def data_gen(t=0):
    cnt = 0
    while cnt < FRAMES:
        cnt += 1
        t += INTERVALS
        yield t, function_to_animate(t)
        

def data2_gen(t=0):
    cnt = 0
    while cnt < FRAMES:
        cnt += 1
        t += INTERVALS
        yield t, np.cos(t)
        
        
def function_to_animate(t):
    return np.sin(t)

    
   

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.grid()
xdata, ydata = [], []


def run(data):
    # update the data
    t, y = data
    xdata.append(t)
    ydata.append(y)
    xmin, xmax = ax.get_xlim()
    
    ymin, ymax = ax.get_ylim()

    if t >= xmax:
        ax.set_xlim(xmin, 2*xmax)
        ax.figure.canvas.draw()
    line.set_data(xdata, ydata)
    
    if t < xmin:
        ax.set_xlim(-2*xmin, xmax)
        ax.figure.canvas.draw()
    line.set_data(xdata, ydata)
    
    
    if y>=ymax:
        ax.set_ylim(ymin, 2*ymax)
        ax.figure.canvas.draw()
    line.set_data(xdata, ydata)
    
    if y<=ymin:
        ax.set_ylim(-2*ymin, ymax)
        ax.figure.canvas.draw()
    line.set_data(xdata, ydata)
        
    return line,

ani = animation.FuncAnimation(fig, func = run, frames = data_gen(), blit=True, interval=INTERVALS,
                      repeat=False, init_func=init)

ani2 = animation.FuncAnimation(fig, func = run, frames = data2_gen(), blit=True, interval=INTERVALS,
                      repeat=False, init_func=init)

plt.show()

"""
fig defines the pyplot window that the animation will take place on
func is a function that will be called at every frame
frames is the source of data that will be passed onto func
Blit priorizes overlapping of multiple animation
Interval is delay between frames in milliseconds
init_func is a function called once that defines the frame of the animation
    
"""
