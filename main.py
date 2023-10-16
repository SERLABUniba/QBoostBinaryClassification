import os
import json

import logging
import logging.handlers
import json
import time
from dwave.system import DWaveSampler, EmbeddingComposite

import pickle
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from qboost import QBoostClassifier, _build_H, _build_bqm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from preprocessing import init_preprocessing

def calculate_matrix(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return accuracy, f1, precision, recall


def reConstructData(predictions, x_test, label):
    # Convert the np array to dataframe
    df_predictions = pd.DataFrame(predictions, columns=[label])

    # Concat dataframes horizontally
    result_data = pd.concat([x_test, df_predictions], axis=1)

    return result_data

def main():
    configuration_file = json.load(open('configuration.json', 'r'))
    car_hacking_path = configuration_file['CarHackingDataset']

    for file in os.listdir(car_hacking_path):
        if '.csv' in file:
            print(f'Processing Dataset: {file[:-4]}')
            dataframe = init_preprocessing(os.path.join(car_hacking_path, file))

            label = dataframe.columns[dataframe.shape[1] - 1]

            X = dataframe.drop(columns=[label])
            y = dataframe[label]

            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            scaler = MinMaxScaler() 

            x_train_scaler = scaler.fit_transform(x_train)
            x_test_scaler = scaler.transform(x_test)
            
            dwave_sampler = DWaveSampler(token="DEV-62189daf614962abecf63bea21714437efe3e084")
            #emb_sampler = EmbeddingComposite(dwave_sampler)
            lmd = 0.1

            load_model = configuration_file['LoadModel']

            # loads models based on the dataset
            if load_model:
                model_dir = configuration_file['SaveModelPath']
                if 'DoS' in file:
                    print('dos')
                    qboost = pickle.load(open(os.path.join(model_dir, 'DoS_dataset_qboost.pickle'), 'rb'))
                elif 'Fuzzy' in file:
                    print('fuzzy')
                    qboost = pickle.load(open(os.path.join(model_dir, 'Fuzzy_dataset_qboost.pickle'), 'rb'))
                elif 'gear' in file:
                    print('gear')
                    qboost = pickle.load(open(os.path.join(model_dir, 'gear_dataset_qboost.pickle'), 'rb'))
                elif 'RPM' in file:
                    print('rpm')
                    qboost = pickle.load(open(os.path.join(model_dir, 'RPM_dataset_qboost.pickle'), 'rb'))
                else:
                    raise Exception('Error when open models!')
            else:
                start = time.perf_counter()
                qboost = QBoostClassifier(x_train_scaler, y_train, lmd)
                end = time.perf_counter()
                train_time = end - start
                print('QBoost training time in seconds :', train_time)

            start = time.perf_counter()
            y_pred = qboost.predict_class(x_test_scaler)
            end = time.perf_counter()
            pred_time = end - start
            print('QBoost prediction time in seconds :', pred_time)
            
            #creation of the matrix of weak classifier predictions
            #H = _build_H(qboost.classifiers, x_test, lmd)

            #Creating the Binary Quadratic Model
            #BQM = _build_bqm(H, y_pred, lmd)

            #emb_sampler.sample(BQM)

            #X_test_array = []

            X_test_array = x_test.to_numpy()

            #print(X_test_array)

            result_data = []

            result_index = [i for i in range(len(y_pred)) if y_pred[i] == -1]
            """for i in range(len(y_pred)):
                if y_pred[i] == -1:
                    result_data.append(X_test_array[i])"""
            
            result_data = x_test.reset_index().loc[result_index, :]

            result_data['D0'] = result_data['D0'].apply( hex ).str[2:]
            result_data['D1'] = result_data['D1'].apply( hex ).str[2:]
            result_data['D2'] = result_data['D2'].apply( hex ).str[2:]
            result_data['D3'] = result_data['D3'].apply( hex ).str[2:]
            result_data['D4'] = result_data['D4'].apply( hex ).str[2:]
            result_data['D5'] = result_data['D5'].apply( hex ).str[2:]
            result_data['D6'] = result_data['D6'].apply( hex ).str[2:]
            result_data['D7'] = result_data['D7'].apply( hex ).str[2:]

            #result_data['D0'] = result_data['D0'].str[2:]

            print(result_data)

            #x_test_reconstruc = pd.DataFrame(x_test, columns = X.columns)

            #result_data = reConstructData(y_pred, X_test_array, label)

            '''
            rows = []
            for _, row in result_data.iterrows():
                row['Event Name'] = 'test'
                row['EventID'] = 'testID'
                row['Vehicle ID'] = 'TESTs'
                row['DATA'] = " ".join([str(row[f'D{i}']) for i in range(8)])
                for i in range(8):
                    del row[f'D{i}']
                rows.append(row.to_dict())

            with open(f'{file}_saveModel Path.json', 'w') as f:
                json.dump(rows , f)
            '''


            
            result_data['ID CAN'] = result_data['CanId'].apply( hex ).str[2:]
            #print(result_data['ID CAN'])
            #result_data['eventName'] = 'TEST NAME'
            result_data['eventCategory'] = 'flow'
            result_data['eventID'] = 'send payload'
            result_data['sourceIP'] = "127.0.0.1"
            result_data['DATA CAN'] = result_data.apply(lambda x: " ".join([str(x[f'D{i}']) for i in range(8)]), axis=1)
            for i in range(8):
                del result_data[f'D{i}']
            del result_data['CanId']

            resultJSON = result_data.to_json(f'{file}.json', orient='records', lines=True)

            #result_data.to_csv('SaveModelPath', mode='a', header=False, index=False)

            # Save the models into a specific directory
            if configuration_file['SaveModel']:
                print('yes')
                model_path_to_save = configuration_file['SaveModelPath']
                model_name = f'{file[:-4]}_qboost.pickle'

                pickle.dump(qboost, open(os.path.join(model_path_to_save, model_name), 'wb'))
                
            accuracy, f1, precision, recall = calculate_matrix(y_test, y_pred)

            print(f'CONFUSION MATRIX')
            C = confusion_matrix(y_test, y_pred)
            total_samples = C.sum()
            C_percent = (C / total_samples) * 100
            TP_percent = C_percent[1, 1]  # Percentuale di veri positivi
            FP_percent = C_percent[0, 1]  # Percentuale di falsi positivi
            TN_percent = C_percent[0, 0]  # Percentuale di veri negativi
            FN_percent = C_percent[1, 0]  # Percentuale di falsi negativi

            print(f'TP: {TP_percent}')
            print(f'FP: {FP_percent}')
            print(f'TN: {TN_percent}')
            print(f'FN: {FN_percent}')



            print(f'TESTING')
            print(f'Accuracy: {accuracy}')
            print(f'F1-Score: {f1}')
            print(f'Precision: {precision}')
            print(f'recall: {recall}')
            print()
            print(classification_report(y_test, y_pred))

            '''accuracy_train, f1_train, precision_train, recall_train = calculate_matrix(y_train, y_pred)

            print(f'TRAINING')
            print(f'Accuracy: {accuracy_train}')
            print(f'F1-Score: {f1_train}')
            print(f'Precision: {precision_train}')
            print(f'recall: {recall_train}')
            print()
            print(classification_report(y_train, y_pred))'''

if __name__ == '__main__':
    main()

