To run this script you will need to download every python file into a directory, access that directory though your shell.

You will need to:
	
	 pip install ale-py 
	 ale-import-roms [path to the folder of the current project that holds all the scripts and the MSPACMAN.bin file]


It is important to note that this path is not the path to the MSPACMAN.bin file, but the path to the directory that holds it.


You can now run the executable Learning.py by running the command "python ./Learning.py" while in the directory of the script. You might have to pip install tensorflow numpy pygame...etc based on the imports you do not already have installed


When running the Learning.py script, you can use arguments to view different things. Most of them are hyperparameters that changes the behaviour of the agent as it is learning through its experiences and its environment. The arguments that are the most important are 'learn' and 'display'. By default, they are learn = False and display = True. If you only want to see the agent's behaviour, keep them as they are. Otherwise, you can change them if you want to further train the agent. 

Finally, the weights_path and epsilon arguments are the one I recommend to use to view the agent's behaviour.

weights_path points to a set of pre-trained weights which are found in the Weights folder in the same directory as the script. Epsilon is a value between 0.01 and 1. It is defaulted to 1, but reducing this value reduces how often the agent makes a random move instead of moving according to its current trained policy.

I suggest to run this script using:

python ./Learning.py --weights_path [absolute_path to one of the weight files] --epsilon 0.5

feel free to experiment with any set of weights.

P.S. There is a CNNLearning and LearningCNN class which are respectively classes that build a model that uses CNN layers to train the model on the visual representation of the game state (as a grid of RGB values) and LearningCNN is a script that does the same as Learning, but slightly modified to agree with the new CNN Model. I have not been able to get meaningful weight results through these classes as they are too computationally expensive for me, but feel free to run them and try to get a better performing agent than I have been able to achieve :).