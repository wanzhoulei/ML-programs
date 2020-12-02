import pandas as pd
from adaboost_2 import AdaBoostClassfier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import numpy as np


def main():
    #upload the data
    df_titanic = pd.read_csv("titanic.csv")
    #drop useless columns
    df_titanic = df_titanic.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'])
    #binarify the sex column
    df_titanic['Sex'] = (df_titanic['Sex'] == 'male')*1
    #fill the lost data in age with mean age
    columns_filled_with_mode = ['Age']
    all_modes = df_titanic[columns_filled_with_mode].mode()  # notice all_modes are in dataframe, and all_modes are in series
    all_modes = all_modes.to_dict(orient="records")[0]  # transfer dataframe to dict
    all_means = df_titanic[columns_filled_with_mode].mean()
    all_means = all_means.to_dict()
    df_titanic.fillna(value=all_modes, inplace=True)
    df_titanic.fillna(value=all_means, inplace=True)
    print(df_titanic)

    X = df_titanic.drop(columns=['Survived'])
    y = df_titanic['Survived']
    #train test split
    x_train, x_test, y_train,y_test=train_test_split(X,y,test_size=0.3)

    '''
    #Hyperparameter visualization
    lr_precision = []
    for i in range(1, 11):
        lr = 0.1*i
        model = AdaBoostClassfier(learning_rate=lr)
        model.fit(x_train, y_train)
        lr_score = model.score(x_test, y_test)
        print(lr, lr_score)
        lr_precision.append(lr_score)
    plt.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], lr_precision, 'k', label= "Precision")
    plt.scatter([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], lr_precision)
    plt.title("Precision Over Different Learning Rate")
    plt.xlabel("Learning Rate")
    plt.ylabel("Precision")
    plt.show()


    #visualization of n_estimator
    n_precision = []
    for n in range(10, 110, 10):
        model = AdaBoostClassfier(n_estimators=n)
        model.fit(x_train, y_train)
        n_score = model.score(x_test, y_test)
        print(n, n_score)
        n_precision.append(n_score)
    plt.plot(range(10, 110, 10), n_precision, 'k')
    plt.show()
    '''

#CV to tune n_estimator 10 folds
#we don't tune learning rate because the visualization shows that lr = 1 gives the best model

    skf = StratifiedKFold(n_splits=10, shuffle=True)

    N_estimator = range(10, 210, 10)
    train_precision = np.zeros((20, 10))
    test_precision = np.zeros((20, 10))
    fold = 0
    for train, test in skf.split(X, y):
        print(fold)
        for n in N_estimator:
            model = AdaBoostClassfier(n_estimators=n)
            model.fit(X.values[train], y[train])
            train_precision[int(n/10 -1), fold] = model.score(X.values[train], y[train]) #X is a dataframe, y is an array
            test_precision[int(n/10 -1), fold] = model.score(X.values[test], y[test])
        fold+=1
    mean_train_precision = train_precision.mean(axis=1)
    mean_test_precision = test_precision.mean(axis=1)
    std_train_precision = train_precision.std(axis=1)
    std_test_precision = test_precision.std(axis=1)

    best_n = np.argmax(mean_test_precision)
    print("Best n_estimator: ", best_n*10 + 10)
    plt.plot(range(10, 210, 10), mean_test_precision, 'k', label = "Test Precision")
    plt.scatter(range(10, 210, 10), mean_test_precision)
    plt.plot(range(10, 210, 10), mean_train_precision, 'k', label = "Train Precision", color = 'orange')
    plt.scatter(range(10, 210, 10), mean_train_precision)
    plt.legend()
    plt.ylabel("Precision")
    plt.xlabel("n_estimator")
    plt.title("Precision vs n_estimator")
    plt.show()


    adaclassifier = AdaBoostClassfier(learning_rate=1, n_estimators=best_n*10 + 10)
    adaclassifier.fit(x_train, y_train)
    prediction_test = adaclassifier.predict(x_test)
    df_result = x_test.copy()
    df_result["Survived"] = y_test
    df_result["Prediction"] = prediction_test
    fig, (ax1, ax2) =  plt.subplots(1,2, figsize=(10,4))
    ax1.scatter(df_result[df_result["Prediction"]==0]["Age"], df_result[df_result["Prediction"]==0]["Fare"], label = "Dead")
    ax1.scatter(df_result[df_result["Prediction"]==1]["Age"], df_result[df_result["Prediction"]==1]["Fare"], label = "Survived")
    ax1.legend()
    ax1.set_xlabel("Age")
    ax1.set_ylabel("Fare Price")
    ax1.set_title("Prediction Result")

    ax2.scatter(df_result[df_result["Survived"]==0]["Age"], df_result[df_result["Survived"]==0]["Fare"], label = "Dead")
    ax2.scatter(df_result[df_result["Survived"]==1]["Age"], df_result[df_result["Survived"]==1]["Fare"], label = "Survived")
    ax2.legend()
    ax2.set_xlabel("Age")
    ax2.set_ylabel("Fare Price")
    ax2.set_title("Reality")
    plt.show()

    print(df_result)
    print("Accuracy: {}".format(adaclassifier.score(x_test, y_test)))
if __name__=='__main__':
    main()
