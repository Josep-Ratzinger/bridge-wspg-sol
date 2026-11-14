"""Máscaras oficiales: superestructura permite viga o tablero, no viga exacta."""
import numpy as np

def semanticbridge(labels):
    mapping={0:0,1:80,2:256,3:128,4:6,5:4,6:8,7:512,8:1}
    return np.array([mapping[int(k)] for k in labels],dtype='i4')

def uav(labels):
 mapping={0:0,1:6,2:1,3:2,4:8}
 return np.array([mapping[int(k)] for k in labels],dtype='i4')
