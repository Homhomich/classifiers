import numpy
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, RocCurveDisplay
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

iris = datasets.load_iris()
X = iris.data
y = iris.target


def choose_feature():
    df = pd.read_excel("datasets/rice_with_eccentricity.xlsx")
    X = df.get(["Ширина", "Высота", "Площадь", "Периметр"]).to_numpy()
    Y = df['Результат'].to_numpy()

    # feature extraction
    test = SelectKBest(score_func=chi2, k=5)
    fit = test.fit(X, Y)
    # summarize scores
    numpy.set_printoptions(precision=3)
    print(fit.scores_)


def random_forest(dataset_path, features_arr):
    df = pd.read_excel(dataset_path)
    X = df.get(features_arr).to_numpy()
    y = df['Результат'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    regressor = RandomForestClassifier(n_estimators=20, random_state=0)
    regressor.fit(X_train, y_train)

    y_pred = np.around(regressor.predict(X_test))
    # print(y_pred)

    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))
    # print(accuracy_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    asc = accuracy_score(y_test, y_pred)

    result = 'confusion matrix \n' + str(cm) + '\n \n' + str(cr) + '\n' + 'accuracy = ' + str(asc)
    return result


# all-rice better
def mlp(dataset_path, features_arr):
    df = pd.read_excel(dataset_path)
    X = df.get(features_arr).to_numpy()
    y = df['Результат'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=2000).fit(X_train, y_train)

    # filename = 'finalized_model.sav' #mlp = pickle.load(open(filename, 'rb')) #pickle.dump(mlp, open(filename, 'wb'))

    predictions = mlp.predict(X_test)

    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(accuracy_score(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)
    cr = classification_report(y_test, predictions)
    asc = accuracy_score(y_test, predictions)

    result = 'confusion matrix \n' + str(cm) + '\n \n' + str(cr) + '\n' + 'accuracy = ' + str(asc)
    return result


# all-rice better
def mlp_test_and_train(dataset_path, dataset_path_test, features_arr):
    df = pd.read_excel(dataset_path)
    X = df.get(features_arr).to_numpy()
    y = df['Результат'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=2000).fit(X_train, y_train)

    predictions = mlp.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(accuracy_score(y_test, predictions))

    df_test = pd.read_excel(dataset_path_test)
    xtest = df_test.get(features_arr).to_numpy()
    xtest = scaler.transform(xtest)
    ytest = df_test['Результат'].to_numpy()

    predictions = mlp.predict(xtest)
    print(confusion_matrix(ytest, predictions))
    print(classification_report(ytest, predictions))
    print(accuracy_score(ytest, predictions))


def random_forest_test_and_train(dataset_path, dataset_path_test, features_arr):
    df = pd.read_excel(dataset_path)
    X = df.get(features_arr).to_numpy()
    y = df['Результат'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    regressor = RandomForestClassifier(n_estimators=20, random_state=0).fit(X_train, y_train)

    predictions = regressor.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(accuracy_score(y_test, predictions))

    df_test = pd.read_excel(dataset_path_test)
    xtest = df_test.get(features_arr).to_numpy()
    xtest = sc.transform(xtest)
    ytest = df_test['Результат'].to_numpy()

    predictions = regressor.predict(xtest)
    print(confusion_matrix(ytest, predictions))
    print(classification_report(ytest, predictions))
    print(accuracy_score(ytest, predictions))


def random_forest_learning_curve(dataset_path, features_arr):
    df = pd.read_excel(dataset_path)
    X = df.get(features_arr).to_numpy()
    y = df['Результат'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    regressor = RandomForestClassifier(n_estimators=20, random_state=0)

    train_sizes, train_scores, validation_scores, fit_times, _ = learning_curve(regressor, X, y, cv=30, return_times=True)
    plt.style.use('seaborn')
    train_scores_mean = -train_scores.mean(axis=1)
    validation_scores_mean = -validation_scores.mean(axis=1)
    plt.plot(train_sizes, train_scores_mean, label='Training error')
    plt.plot(train_sizes, validation_scores_mean, label='Validation error')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title('Learning curves for a linear regression model', fontsize=18, y=1.03)
    plt.legend()
    plt.show()
