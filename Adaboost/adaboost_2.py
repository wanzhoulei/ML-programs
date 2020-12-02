import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from math import*
#define help functions here
def check_data_validation_for_fit(data_x,data_y):
  #notice data_x is np.matrix and data_y is np.array
    if (data_x.dtype == np.dtype('O')):# make sure all data in the matrix is numbers
        raise Exception("String(s) or other non-numerical object(s) in data_X")
    if (data_y.dtype == np.dtype('O')):
        raise Exception("String(s) or other non_numerical object(s) in data_y")
    if (len(data_x.shape) != 2 or len(data_y.shape) != 1):
        raise Exception("Errors in data shape")
    if (data_x.shape[0] != data_y.shape[0]):
        raise Exception("The sample size in data_x doesn't match y vector")
    for each_label in data_y:
        if (each_label!=1 and each_label!=0):
            raise Exception("Only 1 and 0 are allowed in the input y vector")
def check_data_validation_for_predict(classifier,data_x):
    if (data_x.dtype == np.dtype('O')):  # make sure all data in the matrix is numbers
        raise Exception("String(s) or other non-numerical object(s) in data_X")
    if(classifier._feature_number==None):
        raise Exception("Run fit() before running predict()")
    if (len(data_x.shape) >2 or data_x.shape[0]==0):
        raise Exception("Errors in data shape")
    if(len(data_x.shape)==1):
        if(data_x.shape[0] != classifier._feature_number):
            raise Exception("Errors in data shape")
    elif (len(data_x.shape)==2):
        if (data_x.shape[1] != classifier._feature_number):
            raise Exception("Errors in data shape")
#define classifiers here
class StumpClassfier:
    def __init__(self,num_steps=20):
        self._alpha=None
        self._num_steps = num_steps
        self._splitting_dimen=None
        self._splitting_value=None
        self._threshold_ineq=None #choose between "lt" (less than) or "gt" (greater)
        self._feature_number=None
    def fit(self,X,y,weights=[],check_data=True):
        data_x=np.array(X)
        data_y=np.array(y)
        if check_data==True:
            check_data_validation_for_fit(data_x, data_y)
        data_y[data_y == 0] = -1  # transfer 0 to -1 as needed by algorithm
        self._feature_number=data_x.shape[1]
        min_error=float("inf")
        for feature_d in range(data_x.shape[1]):
            value_set_size=len(set(data_x[:,feature_d]))
            #notice here is to optimize and reduce unnecessary num_steps; for example, if a column only has value 0 or 1
            #we do not need to try all possible splitting points from 0 to 1 according to num_steps (or step size),
            #splitting at 0.5 is just fine
            splitting_try = min(value_set_size, self._num_steps)
            #the following process is to reduce the chance that the splitting point is exactly overlapping with a data point
            #especially on max() and min()
            data_range=data_x[:,feature_d].max()-data_x[:,feature_d].min()
            extension=data_range/splitting_try
            step_size=(data_range+extension)/splitting_try
            threshold_value=data_x[:,feature_d].min()-extension/2
            while(threshold_value<=data_x[:,feature_d].max()+extension/2):
                for possible_thresh_ineq in ['lt','gt']:
                    #The following part of code refers to the book Machine Learning in Action by Peter Harrington
                    #and I add some modifications
                    #------------------------------------------
                    predict_array=np.ones(data_x.shape[0])
                    if(possible_thresh_ineq=='lt'):
                        predict_array[data_x[:,feature_d]<=threshold_value]=-1
                    else:
                        predict_array[data_x[:, feature_d] > threshold_value] = -1
                    error_array=np.ones(data_x.shape[0])
                    error_array[predict_array==data_y]=0
                    if(len(weights)==0):
                        weights_vector=np.ones(data_x.shape[0])/data_x.shape[0]
                    else:
                        weights_vector=np.array(weights)
                    weighed_error=sum(weights_vector*error_array)
                    if weighed_error<min_error:
                        min_error=weighed_error
                        self._alpha =0.5*log((1-min_error)/max(min_error,1e-16))
                        self._splitting_dimen = feature_d
                        self._splitting_value = threshold_value
                        self._threshold_ineq = possible_thresh_ineq
                    # ------------------------------------------

                #In the case that all numbers in a column have same value (extension==0), there is dead loop without following break function

                if(extension==0):
                    break
                threshold_value=threshold_value+step_size
    def predict(self,X,check_data=True):
        data_x = np.array(X)
        if check_data==True:
            check_data_validation_for_predict(self, data_x)
        predict_array = np.ones(data_x.shape[0])
        if (self._threshold_ineq == 'lt'):
            predict_array[data_x[:, self._splitting_dimen] <= self._splitting_value ] = 0
        else:
            predict_array[data_x[:, self._splitting_dimen] > self._splitting_value] = 0
        return predict_array
    def score(self,X,y):
        predicted_y=self.predict(X)
        target_y = np.array(y)
        if (predicted_y.shape!=target_y .shape):
            raise Exception("Errors in data shape")
        count=0
        for index in range(predicted_y.shape[0]):
            if predicted_y[index]==target_y[index]:
                count=count+1
        return count/predicted_y.shape[0]
    def update_alpha(self,learning_rate):
        self._alpha=self._alpha*learning_rate
      #print out all the necessary info about the model (to debug)
    def __str__(self):
        return "alpha:{},num_steps:{},splitting_dimen:{},splitting_value:{},threadshold_ineq:{},feature_number:{}".format(
        self._alpha,
        self._num_steps ,
        self._splitting_dimen,
        self._splitting_value,
        self._threshold_ineq,
        self._feature_number)
class AdaBoostClassfier:
    #Initialization functions
    def __init__(self,n_estimators=50, learning_rate=1.0,num_steps=20):
        #public attributes
        if (learning_rate<0 or learning_rate>1):
            raise Exception("Wrong learning rate")
        self._n_estimators=n_estimators
        self._learning_rate=learning_rate
        self._num_steps=num_steps
        self._all_stumps=[]
        self._feature_number=None
    #This part is private functions

    #This part is public functions
    def fit(self,X,y,check_data=True):
        data_x=np.array(X)
        data_y=np.array(y)
        if check_data==True:
            check_data_validation_for_fit(data_x, data_y)
        self._feature_number = data_x.shape[1]
        weights_vector = np.ones(data_x.shape[0]) / data_x.shape[0]
        for i in range(self._n_estimators):
            stump=StumpClassfier(self._num_steps)
            stump.fit(data_x,data_y,weights=weights_vector,check_data=False)
            if(self._learning_rate!=1):
                stump.update_alpha(self._learning_rate)
            self._all_stumps.append(stump)
            predict_result=stump.predict(data_x,check_data=False)
            #update weights vectors for all samples
            for index in range(predict_result.shape[0]):
                if predict_result[index]!=data_y[index]:  #misclassfied
                    weights_vector[index]= weights_vector[index]*(e**(stump._alpha))
                else:  #correctly classified
                    weights_vector[index] = weights_vector[index] * (e ** (-stump._alpha))
            weights_vector=weights_vector/sum(weights_vector)

    def predict(self,X,check_data=True):
        data_x = np.array(X)
        if check_data==True:
            check_data_validation_for_predict(self, data_x)
        predict_array = np.zeros(data_x.shape[0])
        for each_stump in self._all_stumps:
            stump_predict_result=each_stump.predict(data_x,check_data=False)
            stump_predict_result[stump_predict_result==0]=-1
            predict_array=predict_array+stump_predict_result*each_stump._alpha
        predict_array[predict_array>=0]=1
        predict_array[predict_array < 0] = 0
        return predict_array
    def score(self,X,y):
        predicted_y=self.predict(X)
        target_y = np.array(y)
        if (predicted_y.shape!=target_y .shape):
            raise Exception("Errors in data shape")
        count=0
        for index in range(predicted_y.shape[0]):
            if predicted_y[index]==target_y[index]:
                count=count+1
        return count/predicted_y.shape[0]
    #check the situation for all stumps, (mainly to debug)
    def get_stumps(self):
        return self.__all_stumps

    # print out all the necessary info about the model (to debug)
    def __str__(self):
        count = 1
        to_return=""
        for each_stump in self._all_stumps:
            to_return=to_return+"Stump: "+str(count)+"\n"+\
                      "alpha:{},num_steps:{},splitting_dimen:{},splitting_value:{},threadshold_ineq:{},feature_number:{}".format(
        each_stump._alpha,
        each_stump._num_steps ,
        each_stump._splitting_dimen,
        each_stump._splitting_value,
        each_stump._threshold_ineq,
        each_stump._feature_number)+'\n'
            count=count+1
        return to_return
if __name__=='__main__':
    iris=datasets.load_iris()
    X = iris.data
    Y= iris.target
    x_train, x_test, y_train,y_test=train_test_split(X,Y,test_size=0.3)
    y_train[y_train == 2] = 0
    model=AdaBoostClassfier(n_estimators=50, learning_rate=1,num_steps=10)
    #model=StumpClassfier()
    model.fit(x_train,y_train)
    y_test[y_test == 2] = 0
    print(model.score(x_test,y_test))
    print(x_test)
    print(y_test)
    print(model.predict(x_test))
