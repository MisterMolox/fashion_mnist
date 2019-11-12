import numpy as np
from tqdm import tqdm_notebook

def shuffle_data(train, test):
  
  count = np.arange(28)
  new_train = []
  new_test = []
  for i in tqdm_notebook(range(train.shape[0])):
    np.random.shuffle(count)
    new_train.append(train[i,count])

  for i in tqdm_notebook(range(test.shape[0])):
    np.random.shuffle(count)
    new_test.append(test[i,count])

  new_train = np.array(new_train)
  new_test = np.array(new_test)
  return new_train, new_test

def cut_data(x_test, y_test):
  
  count = np.zeros(10) 
  distrib = [10, 150, 100, 400, 900, 50, 1000, 600, 300, 400]

  new_y = []
  new_x = []

  for i in tqdm_notebook(range(10000)):
    
    if count[y_test[i]]<distrib[y_test[i]]:
      
      new_y.append(y_test[i])
      new_x.append(x_test[i])
      count[y_test[i]] += 1

  return new_x, new_y
