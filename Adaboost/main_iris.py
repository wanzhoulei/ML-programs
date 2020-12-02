import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from adaboost_2 import AdaBoostClassfier
from adaboost_2 import StumpClassfier

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)

from sklearn import datasets
from sklearn.model_selection import train_test_split

# define the function that convert the y into a binary 2D matrix
def toMatrix(y):
    matrix = []
    for i in range(len(y)):
        if y[i] == 0:
            matrix.append([1, 0, 0])
        elif y[i] == 1:
            matrix.append([0, 1, 0])
        else:
            matrix.append([0, 0, 1])
    return matrix

def backtoArray(matrix):
    result = []
    for i in range(np.shape(matrix)[0]):
        if matrix[i][0] == 1:
            result.append(0)
        elif matrix[i][1] == 1:
            result.append(1)
        else:
            result.append(2)
    return result

def splitmatrix(matrix):
    y0 = []; y1 = []; y2 = []
    for i in range(np.shape(matrix)[0]):
        y0.append(matrix[i][0])
        y1.append(matrix[i][1])
        y2.append(matrix[i][2])
    return y0, y1, y2

#define the function that calculate the precision of classifiying a specific genre
#precision = TP/(TP + FP)
def precision(y_predict, y_real, num):
    TP_num = 0; TPFP_num = 0;
    for i in range(len(y_predict)):
        if y_predict[i] == num:
            TPFP_num+=1
            if  y_real[i] == num:
                TP_num+=1
    return float(TP_num)/float(TPFP_num)

def averagePrecision(y_predict, y_real):
    result = 0
    for i in range(3):
        result += precision(y_predict, y_real, i)
    return result/3

#define the function that calculates the recall rate of classifying a specific genre
#recall rate = TP/(TP+FN)
def recallRate(y_predict, y_real, num):
    TP_num = 0;
    TPFN_num = 0;
    for i in range(len(y_predict)):
        if y_real[i] == num:
            TPFN_num += 1
            if y_predict[i] == num:
                TP_num += 1
    return float(TP_num)/float(TPFN_num)

def averageRecall(y_predict, y_real):
    result = 0
    for i in range(3):
        result += recallRate(y_predict, y_real, i)
    return result/3

#define the function that takes a dataframe and deals with the situation in which more than one classifiers give positive
#result or all classifiers give negetive result
#the shape of df must be k * 3 and columns name must be 0, 1, 2
def processDF(df, score0, score1, score2):
    df["sum"] = df["0"] + df["1"] + df["2"]
    # find out which is the most/least confident
    scorelist = [score0, score1, score2]
    most_confident = str(0);
    least_confident = str(0);
    most = scorelist[0];
    least = scorelist[0];
    for i in range(len(scorelist)):
        if scorelist[i] > most:
            most = scorelist[i]
            most_confident = str(i)
        if scorelist[i] < least:
            least = scorelist[i]
            least_confident = str(i)
    for i in range(np.shape(df)[0]):
        if df["sum"][i] == 0:  # if all the classifier give negative result, set the least confident result to be 1
            df[least_confident][i] = 1
        elif df["sum"][i] == 3:  # if all classifiers give positive result, set only the most confident to be 1
            df["0"] = 0;
            df["1"] = 0;
            df["2"] = 0;
            df[most_confident] = 1
        elif df["sum"][i] == 2:
            if df["0"][i] == 0:
                if score1 >= score2:
                    df["2"][i] = 0
                else:
                    df["1"][i] = 0
            elif df["1"][i] == 0:
                if score0 >= score2:
                    df["2"][i] = 0
                else:
                    df["0"][i] = 0
            else:
                if score0 >= score1:
                    df["1"][i] = 0
                else:
                    df["0"][i] = 0
    df = df.drop(columns=['sum'])
def main():
    # load the data
    iris=datasets.load_iris()
    X=iris.data
    Y= iris.target
    # split into train and test 70% vs. 30%
    x_train, x_test, y_train,y_test=train_test_split(X,Y,test_size=0.3)

    # convert y_test and y_train in to 2D array
    y_train_matrix = toMatrix(y_train)
    y_test_matrix = toMatrix(y_test)
    #split them
    y0_train, y1_train, y2_train = splitmatrix(y_train_matrix)
    y0_test, y1_test, y2_test = splitmatrix(y_test_matrix)

    #data visualization the relationship between n_estimator and precision
    #it shows that out dataset is so small and simple that the hyper parameter n_estimator is not necessary in our case
    #but it is necessary and useful in more complex and big data
    for list in [range(10, 110, 10), range(1, 11)]:
        precision_list_n_estimator = []
        for N_estimator in list:
            # the first classifier that classifies 0
            model0 = AdaBoostClassfier(n_estimators=N_estimator)
            model0.fit(x_train, y0_train)
            # second classifier
            model1 = AdaBoostClassfier(n_estimators=N_estimator)
            model1.fit(x_train, y1_train)
            #third classifier
            model2 = AdaBoostClassfier(n_estimators=N_estimator)
            model2.fit(x_train, y2_train)
            # build the df that stores the result
            df_ = pd.DataFrame({"0": model0.predict(x_test), "1": model1.predict(x_test), "2": model2.predict(x_test)})
            processDF(df_, model0.score(x_test, y0_test), model1.score(x_test, y1_test), model2.score(x_test, y2_test))
            prediction = backtoArray(df_.values)
            print("n_estimators:", N_estimator)
            print("precision: ", averagePrecision(prediction, y_test))
            precision_list_n_estimator.append(averagePrecision(prediction, y_test))
        plt.plot(list, precision_list_n_estimator, 'k', label= "Precision")
        plt.scatter(list, precision_list_n_estimator)
        plt.title("Precision Over Different n_estimators")
        plt.xlabel("n_estimators")
        plt.ylabel("Precision")
        plt.show()

    #fit a single stump and an ada with single estimator over 3 types of flowers to show that
    #this data is so small and easy that only one stump or few stumps can achieve very high precision
    #over type 0
    stump0 = StumpClassfier()
    stump0.fit(x_train, y0_train)
    print(stump0.score(x_test, y0_test))

    model_0 = AdaBoostClassfier(n_estimators=1)
    model_0.fit(x_train, y0_train)
    print(model_0.score(x_test, y0_test))
    print(model_0)

    #over type 1
    stump1 = StumpClassfier()
    stump1.fit(x_train, y1_train)
    print(stump1.score(x_test, y1_test))

    model_1 = AdaBoostClassfier(n_estimators=1)
    model_1.fit(x_train, y1_train)
    print(model_1.score(x_test, y1_test))
    print(model_1)

    #over type 2
    stump2 = StumpClassfier()
    stump2.fit(x_train, y2_train)
    print(stump2.score(x_test, y2_test))

    model_2 = AdaBoostClassfier(n_estimators=1)
    model_2.fit(x_train, y2_train)
    print(model_2.score(x_test, y2_test))
    print(model_2)


    #data visualization the relationship between learning rate and precision
    #it shows that out dataset is so small and simple that the hyper parameterlearning rate is not necessary in our case
    #but it is necessary and useful in more complex and big data
    precision_list = []
    for i in range(1, 10):
        learning_Rate = i/10
        # the first classifier that classifies 0
        model0 = AdaBoostClassfier(learning_rate=learning_Rate)
        model0.fit(x_train, y0_train)
        # second classifier
        model1 = AdaBoostClassfier(learning_rate=learning_Rate)
        model1.fit(x_train, y1_train)
        #third classifier
        model2 = AdaBoostClassfier(learning_rate=learning_Rate)
        model2.fit(x_train, y2_train)
        # build the df that stores the result
        df_ = pd.DataFrame({"0": model0.predict(x_test), "1": model1.predict(x_test), "2": model2.predict(x_test)})
        processDF(df_, model0.score(x_test, y0_test), model1.score(x_test, y1_test), model2.score(x_test, y2_test))
        prediction = backtoArray(df_.values)
        print(learning_Rate)
        print("precision: ", averagePrecision(prediction, y_test))
        precision_list.append(averagePrecision(prediction, y_test))
    plt.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], precision_list, 'k', label= "Precision")
    plt.scatter([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], precision_list)
    plt.title("Precision Over Different Learning Rate")
    plt.xlabel("Learning Rate")
    plt.ylabel("Precision")
    plt.show()

    #now we fit the model over the real data set
    # fit the train sets
    adaboost0 = AdaBoostClassfier()
    adaboost0.fit(x_train, y0_train)
    adaboost1 = AdaBoostClassfier()
    adaboost1.fit(x_train, y1_train)
    adaboost2 = AdaBoostClassfier()
    adaboost2.fit(x_train, y2_train)

    # make the prediction
    predict_0 = adaboost0.predict(x_test)
    predict_1 = adaboost1.predict(x_test)
    predict_2 = adaboost2.predict(x_test)

    #build the dataframe that stores three predictions
    df = pd.DataFrame({"0" : predict_0, "1" : predict_1, "2" : predict_2})
    processDF(df, adaboost0.score(x_test, y0_test), adaboost1.score(x_test, y1_test), adaboost2.score(x_test, y2_test)) #process the predictions

    #convert the dataframe into a final result of a list containing 0, 1, 2
    final_prediction = backtoArray(df.values)

    #build the dataframe that presents the result
    df_result = pd.DataFrame(x_test, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    df_result["Real_type"] = y_test
    df_result["Prediction"] = final_prediction
    print(df_result)
    print("Note: 0, 1, 2 represents: Setosa, Versicolour, and Virginica respectively.")
    print()

    #print the precision and recall rate for each class
    print("Precision on 0 (Setosa): {}".format(precision(final_prediction, y_test, 0)))
    print("Recall Rate on 0 (Setosa): {}".format(recallRate(final_prediction, y_test, 0)))
    print()
    print("Precision on 1 (Versicolour): {}".format(precision(final_prediction, y_test, 1)))
    print("Recall Rate on 1 (Versicolour): {}".format(recallRate(final_prediction, y_test, 1)))
    print()
    print("Precision on 2 (Virginica): {}".format(precision(final_prediction, y_test, 2)))
    print("Recall Rate on 2 (Virginica): {}".format(recallRate(final_prediction, y_test, 2)))
    print()
    print("Average Precision: {}".format(averagePrecision(final_prediction, y_test)))
    print("Average Recall Rate: {}".format(averageRecall(final_prediction, y_test)))

    #visual presentation of the result

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10,8))
    ax1.scatter(df_result[df_result["Prediction"] == 0]["sepal_length"], df_result[df_result["Prediction"] == 0]["sepal_width"], label= "0 Setosa")
    ax1.scatter(df_result[df_result["Prediction"] == 1]["sepal_length"], df_result[df_result["Prediction"] == 1]["sepal_width"], label = "1 Versicolour")
    ax1.scatter(df_result[df_result["Prediction"] == 2]["sepal_length"], df_result[df_result["Prediction"] == 2]["sepal_width"], label = "2 Virginica")
    ax1.set_title("Predicted Type vs Sepal length and width")
    ax1.set_xlabel("Sepal Length")
    ax1.set_ylabel("Sepal Width")
    ax1.legend()

    ax2.scatter(df_result[df_result["Real_type"] == 0]["sepal_length"], df_result[df_result["Real_type"] == 0]["sepal_width"], label= "0 Setosa")
    ax2.scatter(df_result[df_result["Real_type"] == 1]["sepal_length"], df_result[df_result["Real_type"] == 1]["sepal_width"], label = "1 Versicolour")
    ax2.scatter(df_result[df_result["Real_type"] == 2]["sepal_length"], df_result[df_result["Real_type"] == 2]["sepal_width"], label = "2 Virginica")
    ax2.set_title("Real Type vs Sepal length and width")
    ax2.set_xlabel("Sepal Length")
    ax2.set_ylabel("Sepal Width")
    ax2.legend()

    ax3.scatter(df_result[df_result["Prediction"] == 0]["petal_length"], df_result[df_result["Prediction"] == 0]["petal_width"], label= "0 Setosa")
    ax3.scatter(df_result[df_result["Prediction"] == 1]["petal_length"], df_result[df_result["Prediction"] == 1]["petal_width"], label = "1 Versicolour")
    ax3.scatter(df_result[df_result["Prediction"] == 2]["petal_length"], df_result[df_result["Prediction"] == 2]["petal_width"], label = "2 Virginica")
    ax3.set_title("Predicted Type vs Pedal length and width")
    ax3.set_xlabel("Padel Length")
    ax3.set_ylabel("Padel Width")
    ax3.legend()


    ax4.scatter(df_result[df_result["Real_type"] == 0]["petal_length"], df_result[df_result["Real_type"] == 0]["petal_width"], label= "0 Setosa")
    ax4.scatter(df_result[df_result["Real_type"] == 1]["petal_length"], df_result[df_result["Real_type"] == 1]["petal_width"], label = "1 Versicolour")
    ax4.scatter(df_result[df_result["Real_type"] == 2]["petal_length"], df_result[df_result["Real_type"] == 2]["petal_width"], label = "2 Virginica")
    ax4.set_title("Real Type vs Padel length and width")
    ax4.set_xlabel("Padel Length")
    ax4.set_ylabel("Padel Width")
    ax4.legend()

    plt.show()
if __name__=='__main__':
    main()


