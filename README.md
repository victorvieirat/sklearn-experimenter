
`sklearn_experimenter` is a Python class that allows you to easily run experiments with different datasets, models, splits, and metrics using scikit-learn. It provides a convenient way to organize and automate your machine learning experiments.

---
1. [Last Update](#last-update)
1. [Usage](#usage)
    1. [Add dataset](#add-datasets)
         1. [Custom Dataset Reader](#dataset-reader)       
        1. [Custom feature/target](#features-and-target)
    1. [Add splits](#add-splits)
    1. [Add Models](#add-models)
    1. [Add Metrics](#add-metrics)
    1. [Add Seeds](#add-random-seeds)
1. [Run and Save](#run-experiments-and-save-results)
1. [To do](#todo)
___

## Last Update:
- **New Feature:** : Now it`s possible to create a custom dataset reader and uses sklearn pipelines on add_models().

## Usage

### Importing the Required Modules
```python
from sklearn_experimenter import ModelRunner
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
```

### Instantiate the ModelRunner
```python
runner = ModelRunner()
```

## Add Datasets
To add datasets using the `runner.data.add_datasets()` method, you can follow these examples:

1. Adding a folder path:
```python
runner.data.add_datasets(['bases/'])
```
2. Adding a file path:
```python
runner.data.add_datasets(['bases/df1.csv'])
```
3. Adding a DataFrame with a name:
```python
runner.data.add_datasets([(X, 'X')])
```
4. Adding a DataFrame without a name:
```python
runner.data.add_datasets([X])
```
5. Mixing different types:
```python
runner.data.add_datasets(['bases/','bases/df1.csv',(X,'X'),X])
```
### Dataset Reader:
Is possible to set a custom dataset reader, before adding the dataset. The return of the function should be a pandas DataFrame.
```python
def custom_reader(filename):
    data = pd.read_csv(filename,decimal=',',sep=';')
    return data
#Função personalizada para leitura do dataset
runner.data.reader = custom_reader
```

### Features and Target:

In the current version, the target column is automatically set as the last column in each dataset, and the remaining columns are considered as the feature columns. However, if you need to customize the target and feature columns for a specific dataset, you can use the following approach:

```python
runner.data.datasets[Nth].target = 'target_column_name'
runner.data.datasets[Nth].feature = ['feature_column1', 'feature_column2', ...]
```

Replace `Nth` with the index of the dataset for which you want to modify the target and feature columns. Set `'target_column_name'` to the desired name of the target column, and provide a list of column names `['feature_column1', 'feature_column2', ...]` for the feature columns.

### Add Splits
```python
runner.splits.add_holdout([0.3])  # Holdouts
runner.splits.add_fold([3])       # K-folds
```

### Add Models
It`s possible to add models by name, by the fuction o even add a pipeline as a model.
```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('bagging',  BaggingClassifier(estimator=knn()))
])

runner.models.add_models(['DecisionTreeClassifier', 
                         BaggingClassifier(estimator=knn()),
                         pipeline]
                         )
```

### Add Metrics
```python
runner.metrics.add_score([
    confusion_matrix,
    accuracy_score,
    lambda y_true, y_pred: f1_score(y_true, y_pred, average='micro')
])
```

### Add Random Seeds
```python
runner.random.add_model_seed([42])
```

### Run Experiments and Save Results
```python
runner.save_path ='output.pkl'
runner.run()
```

For more detailed information on the available methods and functionalities of the `sklearn_experimenter` class, please refer to the documentation.

## Todo
- Set n_jobs parameters of functions.
## License
This project is licensed under the MIT License. See the LICENSE file for more details.

