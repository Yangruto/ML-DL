import itertools
from tqdm import tqdm
from sklearn.metrics import accuracy_score

MODEL_PARAMETERS = {
    'class_weight':['balanced'],
    'n_estimators':[100, 200],
    'max_depth' : [10, 100, 1000]
    }

def Grid_Search(model:object, model_parameters:dict, x_train, y_train, x_test, y_test) -> pd.DataFrame:
    """ 
    Use grid search method to find the best hyperparameter combination (by accuracy)
        model: target model
        model_parameters: parameters
        x_train: x training data
        y_train: y training data
        x_test: x testing data
        y_test: y testing data
    """
    parameter_key = model_parameters.keys()
    parameter_list = list(itertools.product(*model_parameters.values()))
    parameter_combinations = []
    for comb in parameter_list:
        tmp = dict()
        for k, j in zip(parameter_key, comb):
            tmp[k] = j
        parameter_combinations.append(tmp)
    # model training
    for i, p in tqdm(enumerate(parameter_combinations), total=len(parameter_combinations)):
        tmp_model = model(**p)
        tmp_model.fit(x_train, y_train)    
        y_pred = tmp_model.predict(x_test)
        parameter_combinations[i]['accuracy'] = accuracy_score(y_test, y_pred)
    return pd.DataFrame(parameter_combinations)