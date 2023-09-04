import os
import json


from dwave.system import DWaveSampler, EmbeddingComposite

import pickle

import pandas as pd

from qboost import QBoostClassifier, _build_H, _build_bqm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

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

            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            dwave_sampler = DWaveSampler()
            emb_sampler = EmbeddingComposite(dwave_sampler)
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
                qboost = QBoostClassifier(x_train, y_train, lmd)

            y_pred = qboost.predict_class(x_test)

            H = _build_H(qboost.classifiers, x_test, lmd)

            BQM = _build_bqm(H, y_pred, lmd)

            sample_set = emb_sampler.sample(BQM)

            x_test = pd.DataFrame(x_test, columns = X.columns)

            result_data = reConstructData(sample_set, x_test, label)

            print(result_data)

            #result_data.to_csv('SaveModelPath', mode='a', header=False, index=False)

            # Save the models into a specific directory
            if configuration_file['SaveModel']:
                print('yes')
                model_path_to_save = configuration_file['SaveModelPath']
                model_name = f'{file[:-4]}_qboost.pickle'

                pickle.dump(qboost, open(os.path.join(model_path_to_save, model_name), 'wb'))
                
            accuracy, f1, precision, recall = calculate_matrix(y_test, y_pred)

            print(f'Accuracy: {accuracy}')
            print(f'F1-Score: {f1}')
            print(f'Precision: {precision}')
            print(f'recall: {recall}')
            print()
            print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    main()

