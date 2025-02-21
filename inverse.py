import numpy as np
import pandas as pd
import itertools
# kharazmi-activeclay-dashboard

def XGenerator(clay, process, n):
    X_description=pd.DataFrame(np.round(np.array(pd.DataFrame(process).describe()), decimals=2))
    generated_colunms= X_description.shape[1]
    # generated_rows= n
    generated_array1=np.empty((n, generated_colunms))
    # generated_array2=np.empty((n, generated_colunms))
    # generated_array3=np.empty((n, generated_colunms))
    # generated_array4=np.empty((n, generated_colunms))
    for i in range(generated_colunms):
        min_= np.array(X_description)[3,i]
        max_= np.array(X_description)[7,i]
        first_quarter= np.array(X_description)[4,i]
        second_quarter= np.array(X_description)[5,i]
        third_quarter= np.array(X_description)[6,i]
        # n= int(generated_rows/4)
        # n=int((n*4-generated_rows)+n)
        a= np.random.uniform(min_, first_quarter,(int(n/4)))
        b= np.random.uniform(first_quarter, second_quarter,(int(n/4)))
        c= np.random.uniform(second_quarter, third_quarter,(int(n/4)))
        d= np.random.uniform(third_quarter, max_,(int(n/4)))
        if i%2==0:
            generated_array1[:,i:i+1]= np.concatenate((a,b,c,d), axis=0).reshape(n,1)
        elif i%3==0:
            generated_array1[:,i:i+1]= np.concatenate((d,c,b,a), axis=0).reshape(n,1)
        elif i%5==0:
            generated_array1[:,i:i+1]= np.concatenate((b,c,d,a), axis=0).reshape(n,1)
        else:
            generated_array1[:,i:i+1]= np.concatenate((d,a,c,b), axis=0).reshape(n,1)
    # n= int(generated_rows/4)
    # n=int((n*4-generated_rows)+n)
    clay = np.array(clay)
    v = np.empty((n, 13))
    for i in range(n):
        v[i]=clay
    v = v.reshape(n , 13)
    # generated_array = np.concatenate((generated_array1,generated_array2,generated_array3,generated_array4), axis=0)
    return(np.concatenate((v,generated_array1), axis=1))


def X_Recommender(clay, model):
    f14= np.arange(1, 5, 0.1)
    f15= np.arange(0.05, 0.2, 0.025)
    f16= np.arange(85, 100, 1)
    f17= np.arange(0.5, 8, 0.25)
    f18= np.arange(100, 300, 20)

    f = [f14, f15, f16, f17, f18]
    process = np.array(list(itertools.product(*f)))
    process = np.unique(process, return_index=False, return_inverse=False, return_counts=False, axis=0)

    n = process.shape[0]
    clay = np.array(clay)
    v = np.empty((n, 13))
    for i in range(n):
        v[i]=clay
    v = v.reshape(n , 13)

    X_pred = np.concatenate((v,process), axis=1)
    X_pred = pd.DataFrame(X_pred, columns = ['Clay MW', 'initial BET (m2/g)', 'd (001) angstrom', 'initial Al2O3', 'initial Fe2O3', 'initial CaO', 'initial MgO', 'initial K2O', 'initial Na2O', 'Octa Oxides Sum', 'Iintra layer Oxides Sum', 'SiO2/Al2O3', 'Acid MW', 'Acid Normal', 'wt clay (g)/ V acid (cc)', 'T(°C)', 'Time (h)', 'highest seen Temp']).drop_duplicates().reindex()
    # X_pred = pd.DataFrame(X_pred, )
    y_pred = model.predict(X_pred)
    # print('Max BET is :', np.amax(y_pred))
    id = np.where(y_pred == np.amax(y_pred))
    # print('Process Params are:', [X_pred[i] for i in id])
    X = [X_pred.iloc[i, :] for i in id]
    y = np.amax(y_pred)
    print(X)

    return y, X