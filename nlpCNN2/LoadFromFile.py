import numpy as np
from sklearn.utils import shuffle

def LoadData(oneHot, t_dataLength, sentenceLength, VectorLength, ValidatePercent):
    import os
    
    Y_ = list()
    Y = list()
    with open("y.bin",'r') as f:
        if(oneHot == 1):
            Y = np.array(f.read().split(' ')[:-1],copy=False,dtype=np.int32)
            Y = np.reshape(Y, [t_dataLength,63])
        if(oneHot == 0):
            for labeldata in f.readlines():
                label = labeldata.split()
                if '1' in label:
                    i = label.index('1')
                    Y_.append(i+1)
                else:
                    Y_.append(0)
            Y = np.array(Y_, dtype=np.int32)
        
   

    #X_ = np.array()
    X = list()
    if(os.path.isfile('./X_np.npy')):
        X = np.load('./X_np.npy')
    else:
        with open("x.bin",'r') as f:
            X = np.fromfile('x.bin',dtype=np.float32, sep=' ')
            X = np.reshape(X,[t_dataLength,sentenceLength,VectorLength,1])
            np.save('./X_np.npy',X)
            
    
    X,Y = shuffle(X,Y,random_state = 0)
    
    assert not np.any(np.isnan(X))
    assert not np.any(np.isnan(Y))
        
    return  X[:int(len(X) * ValidatePercent) ],Y[:int(len(Y) * ValidatePercent)],X[int(len(X) * ValidatePercent):],Y[int(len(Y) * ValidatePercent):]
