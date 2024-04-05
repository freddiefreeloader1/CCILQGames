import numpy as np

def scenerios(scenerio):

    if scenerio == 'overtaking':
        x0s = [[-3.0, -1.0, 0.0], [-3.0, 1.0, 0.0], [-3.0, 0.0, 0.0]]
        xrefs = [[2, 1, 0, 0], [2.0, -1, 0, 0], [3, 0, 0, 0]]
    
    elif scenerio == 'intersection':
        x0s = [[-2.0, -2.0, 0.0], [-2.0, 2.0, 0.0], [0.0, 4.0, -np.pi/2]]
        xrefs = [[3, -2, 0, 0], [3.0, 2, 0, 0], [0, -3, 0, 0]]
    elif scenerio == 'line':
        x0s = [[-3.0, -3.0, 0.0], [-3.0, 3.0, 0.0], [0.0, 0.0, 0.0]]