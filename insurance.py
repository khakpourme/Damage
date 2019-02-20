import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
import numpy as np
import csv
import time
import sys
# import createpoly
# import rasterize

OUTPUT_PATH = "Pdata_2012.csv"
PREDICTION_PATH = "predictiondata.csv"

HEADERS = ["Sum_NOK", "Year", "Days20", "Days-4", "Days-25", "Days.interval", "RR3d", "RR5d", "TAM.Annual",
           "TAMA.Annual", "TAM.Winter", "TAMA.Winter", "TAM.Summer", "TAMA.Summer", "RR.Annual", "RRA.Annual",
           "RR.Winter", "RRA.Winter", "RR.Summer", "RRA.Summer", "TAMA.AnnualST", "TAMA.WinterST", "TAMA.SummerST",
           "RRA.AnnualST", "RRAST.WinterST", "RRA.SummerST", "Navn", "Coast", "SuscAreaKm2", "Pop", "Boligtall",
           "Ad.EMI","AST","CCI","HPI","ConPI","BoligPris","KonsumInnbo", "KPI", "IncomeSurtaxG", "FloodZoneN",
           "FloodZoneArKM2", "KommArea", "CentrScr", "AveAltt", "KommResp", "LivProblms", "ResdArea", "IceArea",
           "BoligAlder", "VUlExpoInd", "lnRixA100", "WindEXP"]

# Split dataset to training and test set and define feature and target variables
def split_dataset(dataset, train_percentage, test_percentage, feature_headers, target_header):

    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header],
                                                        train_size=train_percentage, test_size=test_percentage)
    return train_x, test_x, train_y, test_y

def random_forest_Regressor(features, target):

    clf = RandomForestRegressor()
    clf.fit(features, target)
    return clf

# process confusion matrix
def process_cm(confusion_mat, total, classitem, i=0, to_print=True):

    TP = confusion_mat[i,i]
    FP = confusion_mat[:,i].sum() - TP
    FN = confusion_mat[i,:].sum() - TP
    TN = confusion_mat.sum().sum() - TP - FP - FN
    acc = float((TP+TN) * 100 / total)
    if to_print:
        print('Class ' +classitem+ ' PredictionAcc: {}'.format(acc))
        time.sleep(0.5)
    return acc

def Regression_report_csv(report):

    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[1]
        row['precision'] = float(row_data[2])
        row['recall'] = float(row_data[3])
        row['f1_score'] = float(row_data[4])
        row['support'] = float(row_data[5])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    classlist = list(dataframe['class'])
    return classlist

def save_to_file(testy, pr):

    actual_class = []
    predicted_class = []
    for i in range(testy.count()):
        actual_class.append(list(testy)[i])
        predicted_class.append(pr[i])
    pre_dataframe = pd.DataFrame(actual_class, columns=['Actual_Class'])
    pre_dataframe['Predicted_Value'] = predicted_class
    pre_dataframe.to_csv('regPrediction.csv')
    print 'File regPrediction.csv Saved.'

def training_model():

    dataset = pd.read_csv(OUTPUT_PATH)
    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, 0.3, HEADERS[1:], HEADERS[0])
    print 'Training on file {}'.format(OUTPUT_PATH), 'train set'
    text = '....'
    for char in text:
        print char
        time.sleep(2)
    print('Training done...')
    print 'Testing on file {}'.format(OUTPUT_PATH), 'test set'
    for char in text:
        print char
        time.sleep(2)
    print('Testing done...')
    trained_model = random_forest_Regressor(train_x, train_y)
    predictions = trained_model.predict(test_x)
    report = regression_report(test_y, predictions)
    classlist = regression_report_csv(report)

    print "Total Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x))
    print "Total Test Accuracy  :: ", accuracy_score(test_y, predictions)
    print ""
    time.sleep(2)
    cm = confusion_matrix(test_y, predictions)
    print('Calculating Prediction Accuracy for Sum_NOK:')
    for i in range(len(classlist)):
        process_cm(cm, test_y.count(), classlist[i], i, to_print=True)
    print ""
    save_to_file(test_y, predictions)
    prediction_step(trained_model)

def prediction_step(trained_model):

    print 'Predicting...'
    newdataset = pd.read_csv(PREDICTION_PATH, index_col=0)
    pointCol = newdataset['Point']
    newdataset = newdataset.drop(['Point'], axis=1)
    newpredictions = trained_model.predict(newdataset)
    pred = pd.DataFrame(newpredictions)
    pred["Point"] = pointCol
    pred.columns = ['Sum_NOK', 'Point']
    pred.to_csv("nokprediction.csv")
    # pointshapefilepath = createpoly.create_point_shape(pred)
    # rasterize.rasterize(pointshapefilepath)

def main():

    training_model()

if __name__ == "__main__":

    main()