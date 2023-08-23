# QBoost Classifier
The directory is composed by:

 - models/ : contains the pre-trained QBoost classifiers.
 - configuration.json: contains differents fields to load the datasets, and so on.
 - preprocessing.py: contains the differents methods to pre-process the datasets.
 - main.py: is the principal file.

## configuration.json

It's a JSON file that contains differents fields.

> {  
>     "CarHackingDataset": "PATH",     
>     "SaveModelPath": "PATH",     
>     "LoadModel": true,     
>     "SaveModel": false   
>     }

 - CarHackingDataset: is the ABSOLUTE path where the datasets are contained ('DoS_dataset.csv', ...).
 - SaveModelPath: is the ABSOLUTE path where the models will be saved.
 - LoadModel: boolean if you want to load the models without perform the fit.
 - SaveModel: boolean if you want to save the fitted models.

## Car-Hacking Dataset
The datasets can be downloaded on this site [Car-Hacking Dataset for the intrusion detection](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset), or directly [Dropbox](https://www.dropbox.com/sh/evlfzrac4vipo12/AAAGoF-KutOGVdqNosIZu7XTa?dl=0).

## D-Wave connection
The information about D-Wave to install the pip package and set-up the token can be found here [D-Wave Installation](https://docs.ocean.dwavesys.com/en/latest/overview/install.html).