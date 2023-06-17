from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
from sklearn.base import BaseEstimator
from sklearn import metrics
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd
import os
import inspect


estimators_dict = {item[0]: item[1] for item in all_estimators(type_filter='classifier')}
metrics_dict={
      "accuracy": [metrics.accuracy_score,{}]
      ,"acc": [metrics.accuracy_score,{}]
      ,"balanced_accuracy": [metrics.balanced_accuracy_score,{}]
      ,"top_k_accuracy": [metrics.top_k_accuracy_score,{}]
      ,"average_precision": [metrics.average_precision_score,{}]
      ,"neg_brier_score": [metrics.brier_score_loss,{}]
      ,"f1": [metrics.f1_score,{}]
      ,"f1_micro": [metrics.f1_score,{'average':'micro'}]
      ,"f1_macro": [metrics.f1_score,{'average':'macro'}]
      ,"f1_weighted": [metrics.f1_score,{'average':'weighted'}]
      ,"f1_samples": [metrics.f1_score,{'average':'samples'}]
      ,"neg_log_loss": [metrics.log_loss,{}]
      ,"precision": [metrics.precision_score,{}]
      ,"recall": [metrics.recall_score,{}]
      ,"jaccard": [metrics.jaccard_score,{}]
      ,"roc_auc": [metrics.roc_auc_score,{}]
      ,"roc_auc_ovr": [metrics.roc_auc_score,{'multi_class':'ovr'}]
      ,"roc_auc_ovo": [metrics.roc_auc_score,{'multi_class':'ovo'}]
      ,"roc_auc_ovr_weighted": [metrics.roc_auc_score,{'multi_class':'ovr','average':'weighted'}]
      ,"roc_auc_ovo_weighted": [metrics.roc_auc_score,{'multi_class':'ovo','average':'weighted'}]
      ,"matrix": [metrics.confusion_matrix,{}]
      ,"confusion_matrix": [metrics.confusion_matrix,{}]
    }


def isList(value):
    return isinstance(value, list)
def isString(value):
    return isinstance(value, str)
def isDataset(value):
    return  isinstance(value, pd.DataFrame)
def isModel(value):
    try:
        value.get_params
        return True
    except:
        return False
def is_sklearn_metric(variable):
    return callable(variable) and getattr(metrics, variable.__name__, None) is variable
def is_probalistic(model):
    args = inspect.getfullargspec(model.__init__).kwonlyargs
    return "random_state" in args
def validate(y_true,y_pred, metrics):
    output = {}
    for metric in metrics.score:
        output[metric[0].__name__] = metric[0](y_true=y_true,y_pred=y_pred,**metric[1])
    return output
def get_variable_name(var):
    for name, value in globals().items():
        if id(value) == id(var):
            return name
    return None
"""
=========================================
============= Model Manager =============
=========================================
"""
class ModelManager:
    def __init__(self,models):
        self.models = []
        if models is not None:
            self.add_models(models)
    def __str__(self):
        return ', '.join([item.__class__.__name__ for item in self.models])
    def _process_model_item(self,item):
        if isString(item):
            try:
                model = estimators_dict[item]()
                self.models.append(model)
            except:
                raise ValueError(f'An error occurred while attempting to retrieve the model. Look at {item}')

        elif isModel(item):
            
            self.models.append(item)
        else:
             raise ValueError(f'An error occurred while attempting to retrieve the model. Look at {item}')
        return
    def add_models(self,models):     
        if isList(models):
            for item in models:
                self._process_model_item(item)
        else:
            self._process_model_item(models)
        return 

"""
=========================================
============ Dataset manager ============
=========================================
"""
class DatasetItem:
    def __init__(self,dataset=None,file_path=None):
        
        self.dataset = dataset
        self.target = self.dataset.columns[-1]
        self.feature = self.dataset.columns[:-1]
        self.name = file_path
    def get_target(self):
        return self.dataset[self.target]
    def get_feature(self):
        output = self.dataset[self.feature].copy()
        string_columns = output.select_dtypes(include='object').columns
        output = pd.get_dummies(output, columns=string_columns)
        return output






class DatasetManager:
    def __init__(self,datasets):
        self._count = 0
        self.datasets = []
        if datasets is not None:
            self.add_datasets(datasets)
    def _process_dataset_item(self,item):
        if isString(item):
            if os.path.isdir(item):
                # Iterate over all files in the folder
                for file_name in os.listdir(item):
                    file_path = os.path.join(item, file_name)
                    if os.path.isfile(file_path):
                        self._process_dataset_item(file_path)
            else:
                try:
                    self.datasets.append(DatasetItem(pd.read_csv(item),item))
                except:
                    raise ValueError(f'An error occurred reading dataset. Look at {item}')

        elif isDataset(item):
            name = get_variable_name(item)
            if name is None:
                name = f'Dataset{self._count}'
                self._count += 1
            self.datasets.append(DatasetItem(item,name))
        else:
            raise ValueError(f'An error occurred reading dataset. Look at {item}')
        return
    def add_datasets(self,datasets):     
        if isList(datasets):
            for item in datasets:
                self._process_dataset_item(item)
        else:
            self._process_dataset_item(datasets)
        return 
"""
=========================================
============ Metrics Manager ============
=========================================
"""
class MetrictManager:
    def __init__(self,score):
        self.score = []
        if score is not None:
            self.add_score(score)
        
    def __str__(self):
        return ', '.join([item[0].__name__  for item in self.score])
    def _process_score_item(self,item):
        if isString(item):
                try:
                    self.score.append(metrics_dict[item])
                except:
                    raise ValueError(f"The metric {item} wanst recognized.")

        elif is_sklearn_metric(item):
            self.score.append([item,{}])
        else:
            raise ValueError(f"The metric {item} wanst recognized.")
        return
    def add_score(self,scores):     
        if isList(scores):
            for item in scores:
                self._process_score_item(item)
        else:
            self._process_score_item(scores)
        return 
    
"""
=========================================
============ Split Manager ============
=========================================
"""
class SplitManager:
    def __init__(self,splits_input):
        self.splits = {'holdout':[],
                       'fold':[]}
        if splits_input is not None:
            self.add_splits(splits_input) 
    
    def add_splits(self,splits_input):
        if 'holdout' in  splits_input:
            self.splits['holdout'] += splits_input['holdout']
        if 'fold' in  splits_input:
            self.splits['fold'] += splits_input['fold']
    def add_holdout(self,hold_input):
        if isList(hold_input):
            self.splits['holdout'] += hold_input
        else:
            self.splits['holdout'].append(hold_input)
    def add_fold(self,fold_input):
        if isList(fold_input):
            self.splits['fold'] += fold_input
        else:
            self.splits['fold'].append(fold_input)

"""
=========================================
============ Seed Manager ============
=========================================
"""
class SeedManager:
    def __init__(self,seed_input):
        self.seed = {'model':[],
                     'split':[]}
        if seed_input is not None:
            self.add_seed(seed_input)   
    
    def add_seed(self,seed_input):
        if 'model' in  seed_input:
            self.seed['model'] += seed_input['model']
        if 'split' in  seed_input:
            self.seed['split'] += seed_input['split']
    def add_model_seed(self,model_input):
        self.seed['model'] = []
        if isList(model_input):
            self.seed['model'] += model_input
        else:
            self.seed['model'].append(model_input)
    def add_split_seed(self,split_input):
        self.seed['split'] = []
        if isList(split_input):
            self.seed['split'] += split_input
        else:
            self.seed['split'].append(split_input)

"""
=========================================
============== Model runner =============
=========================================
"""
class ModelRunner: 
    def __init__(self, data=None, metrics_input=None, models=None,splits=None,seeds={'model':[42],'split':[42]}):
        self.models = ModelManager(models)
        self.data = DatasetManager(data)
        self.metrics = MetrictManager(metrics_input)
        self.splits = SplitManager(splits)
        self.random = SeedManager(seeds)
        self.output = None
    def summarize(self):
        print(f'Number of dataframes: {len(self.data.datasets)}')
        print(f'Metrics used: {self.metrics}')
        print(f'Split method:')
        print(f'     - Holdouts: {self.splits.splits["holdout"]}')
        print(f'     - Folds: {self.splits.splits["fold"]}')
        print(f'Seeds methods:')
        print(f'     - Each model will run {len(self.random.seed["model"])} times witch seeds: {self.random.seed["model"]}')
        print(f'     - Each split will run {len(self.random.seed["split"])} times witch seeds: {self.random.seed["split"]}')
        print(f'Models used: {self.models}')
    def _model_runner(self,model_item,new_line, X_train, X_test, y_train, y_test):
        if is_probalistic(model_item):
            for val_model_seed in self.random.seed["model"]:
                new_line['model_seed'] = val_model_seed
                model = model_item
                model.random_state = 42
                model.fit(X_train,y_train)

                new_line_metrics = {}
                new_line_metrics.update(new_line)
                new_line_metrics.update(validate(model.predict(X_test),y_test,self.metrics))
                self.results.append(new_line_metrics)
                if self.save_each_evaluation:
                    pd.DataFrame(self.results).to_pickle("result.pkl")
        else:
            new_line['model_seed'] = np.nan
            model = model_item
            model.fit(X_train,y_train)
            new_line_metrics = {}
            new_line_metrics.update(new_line)
            new_line_metrics.update(validate(model.predict(X_test),y_test,self.metrics))
            self.results.append(new_line_metrics)
            if self.save_each_evaluation:
                pd.DataFrame(self.results).to_pickle("result.pkl")
    def run(self,save_each_evaluation=True):
        self.save_each_evaluation = True
        # Run split
            #Run each model
                # Run model n times
        self.results = []
        new_line = {'df_name':None,
                    'model_name':None,
                    'model_params':None,
                    'split_type':None,
                    'split_value':None,
                    'model_seed':None,
                    'split_seed':None}
        for df in self.data.datasets:
            new_line['df_name'] = df.name
            for model_item in self.models.models:
                new_line['model_name'] = model_item.__class__.__name__
                new_line['model_params'] = model_item.get_params()
                for split_type in list(self.splits.splits.keys()):
                    for split_value in self.splits.splits[split_type]:
                        new_line['split_value'] = split_value
                        for val_split_seed in self.random.seed["split"]:
                            new_line['split_seed'] = val_split_seed
                            if split_type == 'holdout':
                                new_line['split_type'] = split_type
                                X_train, X_test, y_train, y_test = train_test_split(df.get_feature(),
                                                                                    df.get_target(),
                                                                                    test_size=split_value,
                                                                                    random_state=val_split_seed
                                                                                    )
                                
                                self._model_runner(model_item,new_line, X_train, X_test, y_train, y_test)
                            else:
                                kf = KFold(n_splits=split_value, shuffle=True, random_state=val_split_seed)
                                count=1
                                X = df.get_feature()
                                y = df.get_target()
                                for train_index, test_index in kf.split(df.get_feature()):
                                    new_line['split_type'] = f'{split_type}_{count}/{split_value}'
                                    X_train, X_test = X.loc[train_index], X.loc[test_index]
                                    y_train, y_test = y.loc[train_index], y.loc[test_index]
                                    self._model_runner(model_item,new_line, X_train, X_test, y_train, y_test)
        self.results = pd.DataFrame(self.results)
                                    
                                        