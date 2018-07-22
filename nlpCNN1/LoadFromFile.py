import numpy as np
from sklearn.utils import shuffle

def LoadData(t_dataLength = 10188, sentenceLength = 32, VectorLength = 300, ValidatePercent = 0.8):
    print("Loading Data...")
    
    oneHot = 0
    Y_ = list()
    Y = list()
    with open("y.txt") as f:
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
        
    print(len(Y))
    print('Y_FILE READ DONE')

    #X_ = np.array()
    with open("x.bin",'r') as f:
       X = np.fromfile('x.bin',dtype=np.float32, sep=' ')
       X = np.reshape(X,[t_dataLength,sentenceLength,VectorLength,1])
       
       print(len(X),len(X[t_dataLength - 1]),len(X[t_dataLength - 1][sentenceLength - 1]))
       print("X_FILE READ DONE")
    
    print('LoadingNP_array...\n')
    
    X,Y = shuffle(X,Y,random_state = 0)
    
    assert not np.any(np.isnan(X))
    assert not np.any(np.isnan(Y))
        
    return  X[:int(len(X) * ValidatePercent) ],Y[:int(len(Y) * ValidatePercent)],X[int(len(X) * ValidatePercent):],Y[int(len(Y) * ValidatePercent):]
