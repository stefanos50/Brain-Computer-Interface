import numpy as np
import matplotlib.pyplot as plt

def dmatrix(X):
    m,n = X.shape
    a = np.matmul(X , X.T)
    e = np.ones(shape=(m,m))

    d = np.matmul(np.diag(np.diag(a)),e) + np.matmul(e,np.diag(np.diag(a)))-2*a
    return d

def snr_sample(X):
    N,p = X.shape
    average = np.mean(X,axis=0)
    d = dmatrix(X)
    v = np.ones(shape=(1,N))

    NP = (1/(2*p*N*(N-1))) * np.matmul(np.matmul(v,d),v.T)[0][0]
    SP = (1/p) * np.matmul(average , average.T) - (1/N)* NP


    return NP,SP


def create_plot(size,data_X,data_Y,titles,grid_pos,x_labels,y_labels):
    figure, axis = plt.subplots(size[0], size[1])
    for i in range(len(data_X)):
        if size[0] == 1 or size[1] == 1:
            axis[grid_pos[i]].plot(data_X[i], data_Y[i])
            axis[grid_pos[i]].set_title(titles[i])
            axis[grid_pos[i]].set_xlabel(x_labels[i])
            axis[grid_pos[i]].set_ylabel(y_labels[i])
        else:
            axis[grid_pos[i][0], grid_pos[i][1]].plot(data_X[i], data_Y[i])
            axis[grid_pos[i][0], grid_pos[i][1]].set_title(titles[i])
    plt.show()

def signal_to_trials(signal,markers,prest,post):
    trials = []
    for i in range(0,len(markers)-1):
        trial1 = signal[markers[i][0]-prest-1:markers[i][0]+post-1,:]
        trials.append(trial1.flatten())

    return np.array(trials),np.array(list(range(-prest+1, post+1)))

def get_matrix_with_step(matrix,step):
    res = []
    res.append(matrix[0,:])
    for i in range(1,len(matrix)):
        if i % step == 0:
            res.append(matrix[i,:])
    return np.array(res)
