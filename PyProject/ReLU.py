from pprint import pprint

def ReLU_new(x):
    y = x
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            #print('<< x >>')
            #pprint(x[i][j])
            if any(item <= 0 for item in x[i][j]):
                for n in range(x.shape[2]):
                    y[i][j][n] = 0
            #print('<< y >>')
            #pprint(y[i][j])
    return y
