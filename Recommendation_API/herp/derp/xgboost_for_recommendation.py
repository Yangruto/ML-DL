import joblib
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.stats as ss
from typing import Union              

import xgboost
from xgboost import XGBClassifier, plot_importance, plot_tree 

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({'font.size': 16})
mpl.rcParams['figure.figsize'] = (12, 8)

BUSINESS_UNIT = ''
ALL_CAMPAIGNS = []
TRAINING_CAMPAIGNS = []

FEATUE_TEMPLATE = []

MODEL_FEATURES = [i for i in filter(lambda x: (BUSINESS_UNIT in x or x in ('recency')), FEATUE_TEMPLATE)]

PRODUCT_MAPPING_TABLE = {}

# parameter for gridsearch
MODEL_PARAMETERS = {
    'scale_pos_weight':[3, 6],  # use when data unbalanced, weighted on positive ground truth
    'n_estimators':[100, 200], # number of boosting rounds
    'learning_rate':[0.1, 0.3], # learning rate
    # 'gamma':[0, 10, 100], # the mini loss reduction when leaf want to split
    'gamma':[0, 10, 100], # the mini loss reduction when leaf want to split
    # 'max_depth':[3, 6, 8, 10], # max tree depth
    'max_depth':[5, 10, 20], # max tree depth
    'min_child_weight':[1, 5, 10], # when sum of all weights in the leaf is less than min_child_weight, then stop splitting
    'use_label_encoder':[False],
    'eval_metric':['error']
    }

class Recommendation():
    """
    Build a xgboost recommendation model
    """
    def __init__(self) -> None:
        self.FEATUE_TEMPLATE = FEATUE_TEMPLATE
        self.MODEL_FEATURES = MODEL_FEATURES
        self.MODEL_PARAMETERS = MODEL_PARAMETERS
        self.user_data = self._get_user_data()
    
    def training_setting(self):
        """
        Load training data
        """
        self.ALL_CAMPAIGNS = ALL_CAMPAIGNS
        self.TRAINING_CAMPAIGNS = TRAINING_CAMPAIGNS
        self.training_data = self._get_training_data()

    def _get_user_data(self) -> pd.DataFrame:
        """
        Read User x Product data for training or recommendation
        """
        data_for_recommend = pd.read_csv('./data/user_product.csv')
        member_data = pd.read_csv('./data/member_data.csv', usecols=['Name', 'email', 'epaper'])
        training_data = data_for_recommend.merge(member_data, how='left', on='email')
        training_data = training_data.loc[training_data.email.notnull()]
        training_data['epaper'] = training_data['epaper'].astype('str')
        return training_data
    
    def _get_training_data(self) -> pd.DataFrame:
        """
        Read EDM results and process edm data
        """
        data = pd.DataFrame()
        for i in self.ALL_CAMPAIGNS:
            tmp_sent = pd.read_csv(f'./edm/{i}_NOT_OPEN.csv', names=['name', 'email', 'band', 'open_email'], header=0)
            tmp_open = pd.read_csv(f'./edm/{i}_OPEN.csv', names=['name', 'email', 'band', 'open_email'], header=0)
            tmp_sent['open_email'] = 0
            tmp_open['open_email'] = 1
            tmp_data = pd.concat([tmp_sent, tmp_open])
            tmp_data['edm_name'] = i
            data = data.append(tmp_data)
        data = data.merge(self.user_data, how='left', on='email')
        data = data.loc[data.recency.notnull()]
        return data

    def fill_null_data(self) -> None:
        """
        Fill null values through interativeimputer 
        """
        imp_mean = IterativeImputer(estimator=BayesianRidge())
        imp_mean.fit(self.user_data)
        self.user_data = imp_mean.transform(self.user_data)

    def set_validate_data(self) -> None:
        """
        Choose specific campaign(s) as validation data and shuffle the remains as training data
        """
        self.validate_data = self.training_data.loc[~self.training_data.edm_name.isin(self.TRAINING_CAMPAIGNS)]
        self.training_data = self.training_data.loc[self.training_data.edm_name.isin(self.TRAINING_CAMPAIGNS)]
        self.training_data = self.training_data.sample(frac=1)

    def split_data(self, test_size:float) -> None:
        """
        Split training data to training dataset and tesing dataset
        """
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.training_data.loc[:, MODEL_FEATURES],    
            self.training_data['open_email'], 
            test_size=test_size, 
            random_state=42, 
            stratify=self.training_data['open_email']
        )
    
    def get_scale_weight(self) -> None:
        """
        Calculate the number of negative and positive 
        """
        self.scale_weight = len(self.y_train.loc[self.y_train==0]) / self.y_train.sum()
        self.MODEL_PARAMETERS['scale_pos_weight'].append(self.scale_weight)

    def build_model(self) -> None:
        """
        Build xgboost model
        """
        model = XGBClassifier(
            scale_pos_weight=self.scale_weight,
            n_estimators=200,
            learning_rate=0.1,
            gamma=0,
            max_depth=20, 
            min_child_weight=5, 
            use_label_encoder=False,
            eval_metric='error'
        )
        self.model = model
    
    def save_model(self, path) -> None:
        joblib.dump(self.model, path)
        
    def precision_recall(self, ground_truth: pd.Series, predict: np.array) -> None:
        """
        Calculate the precision, recall and accuracy of the test data
            ground_truth: the ground gruth of the test data
            predict: the predicted result of the test data
        """
        precision, recall, fscore, support = score(ground_truth, predict)
        result = pd.DataFrame([precision, recall, fscore, support])
        result = result.transpose()
        result.columns=['precision','recall','fscore', 'support']
        print(result)
        print(f'Accuracy: {accuracy_score(ground_truth, predict) * 100}%')

    def validate_model(self, plot_result:bool) -> Union[int, int, int]:
        """
        Validate the performance of the model
            plot_result: wheather to plot the validation result 
        """ 
        open_ratio = len(self.validate_data.loc[self.validate_data.open_email==1]) / len(self.validate_data) * 100
        sample_size = len(self.validate_data)
        validate_result = self.validate_data.copy()
        validate_result['prob'] = self.model.predict_proba(self.validate_data[self.MODEL_FEATURES])[:, 1]
        validate_result = validate_result[['prob', 'open_email']]
        validate_result.columns=['prob', 'ground_truth']
        validate_result = validate_result.sort_values('prob', ascending=False)
        validate_result['cum'] = validate_result.ground_truth.cumsum()
        validate_result = validate_result.reset_index(drop=True)
        validate_result['member_number'] = validate_result.index + 1
        validate_result['ratio'] = round(validate_result['cum'] / validate_result['member_number'] * 100, 2)
        validate_result['prob'] = validate_result['prob'] * 100
        validate_result['ratio_diff'] = validate_result['ratio'] - open_ratio
        validate_result['performance'] = np.where(validate_result['ratio_diff'] > 0, 1, 0)
        performance = round(sum(validate_result['performance']) / len(validate_result) * 100, 2)
        increase_ratio = round(validate_result['ratio_diff'].mean(), 2)
        accuracy = accuracy_score(self.validate_data['open_email'], self.model.predict(self.validate_data.loc[:, MODEL_FEATURES]))
        # print(campaign_name)
        print(f'Sample size is {sample_size}')
        print(f'Original open rate is {round(open_ratio, 2)}%')
        print(f'The probability of the model better than the original is {performance}%.')
        print(f'The open email ratio of the model increase {increase_ratio}%.')
        if plot_result == True:
            plt.figure(dpi=150)
            sns.lineplot(x=validate_result['member_number'], y=validate_result['ratio'], label='accululate_open_rate')
            sns.lineplot(x=validate_result['member_number'], y=validate_result['prob'], label='individual_open_rate_predicted')
            plt.axhline(validate_result.loc[len(validate_result) - 1, 'ratio'], ls='--', color='red')
            # plt.axhline(50, ls='--', color='green')
            plt.xlabel('customer_number')
            plt.ylabel('individual_open_rate_predicted / accululate_open_rate')
            # plt.title(campaign_name)
            plt.savefig(f'./data/image/result.png')
        return performance, increase_ratio, accuracy

    def grid_search(self) -> None:
        """ 
        Use grid search method to find the best hyperparameter combination
        """
        parameter_key = self.MODEL_PARAMETERS.keys()
        parameter_list = list(itertools.product(*self.MODEL_PARAMETERS.values()))
        parameter_combinations = []
        for comb in parameter_list:
            tmp = dict()
            for k, j in zip(parameter_key, comb):
                tmp[k] = j
            parameter_combinations.append(tmp)
        # model training
        for i, p in tqdm(enumerate(parameter_combinations)):
            self.model = XGBClassifier(**p)
            self.model.fit(self.x_train, self.y_train)    
            performance, increase_ratio, accuracy = self.validate_model(False)
            parameter_combinations[i]['performace'] = performance
            parameter_combinations[i]['increase_ratio'] = increase_ratio
            parameter_combinations[i]['accuracy'] = accuracy
        self.parameter_combinations = pd.DataFrame(parameter_combinations)
        print(f'There are {len(parameter_combinations)} different hyperparameter combinations.')

    def set_best_hyperparameter(self) -> xgboost.sklearn.XGBClassifier:
        """
        Set the best hyperparameters to train the best model
        """
        increase_ratio = self.parameter_combinations.loc[:, self.parameter_combinations.columns.str.contains('increase_ratio')].mean(axis=1)   
        accuracy = self.parameter_combinations.loc[:, self.parameter_combinations.columns.str.contains('accuracy')].mean(axis=1)
        weighted_result = ss.rankdata(increase_ratio) * 0.7 + ss.rankdata(accuracy) * 0.3
        best_parameter = self.parameter_combinations.loc[weighted_result.argmax(), list(self.MODEL_PARAMETERS.keys())].to_dict()
        print(f'Best hyperparameter: {best_parameter}')
        self.model = XGBClassifier(**best_parameter)
        self.model.fit(self.x_train, self.y_train)

    def load_model(self, path) -> None:
        """
        Load model
            path: model path
        """
        self.model = joblib.load(path)

    def edm_recommend(self, method='threshold', max_recency=1000, min_open_prob=50, headcount=5000):
        """
        Recommend a user list for EDM. There are two methods you can apply.\ 
        One is recommend by threshold (recnecy and open probability).\ 
        The other is by headcount, how many users you want to recommend.
            method: 'threshold' or 'headcount'
            max_recency: filter out if their recency is larger than the threshold you set (for threshold method)
            min_open_prob: filter out whoose open probability is lower than the threshold you set (for threshold method)
            headcount: how many users you want to recommend (for headcount method)
        """
        recommend_list = self.user_data.loc[self.user_data.epaper=='1']
        recommend_list.loc[:, ['not_open_prob', 'open_prob']] = self.model.predict_proba(recommend_list[self.MODEL_FEATURES]) * 100
        recommend_list = recommend_list.sort_values('open_prob', ascending=False)
        if method == 'threshold':
            print(f'You are applying threshold method: max_recency is {max_recency}, min_open_prob is {min_open_prob}')
            recommend_list = recommend_list.loc[(recommend_list.recency < max_recency) & (recommend_list.open_prob > min_open_prob), ['Name', 'email'] + self.MODEL_FEATURES]
        else:
            print(f'You are applying headcount method: headcount is {headcount}')
            recommend_list = recommend_list.loc[(recommend_list.recency < max_recency), ['Name', 'email'] + self.MODEL_FEATURES].head(headcount)
        product_interest = self.MODEL_FEATURES.copy()
        product_interest.remove('recency')
        recommend_list['Interested Product'] = recommend_list[product_interest].idxmax(axis="columns")
        recommend_list['Interested Product'] = recommend_list['Interested Product'].map(PRODUCT_MAPPING_TABLE)
        recommend_list = recommend_list[['Name', 'email', 'Interested Product']]
        return recommend_list
        
if __name__ == "__main__":
    # build model
    recommendation = Recommendation()
    recommendation.training_setting()
    recommendation.set_validate_data()
    recommendation.split_data(0.01)
    recommendation.get_scale_weight()
    recommendation.build_model()
    recommendation.model.fit(recommendation.x_train, recommendation.y_train)
    y_pred = recommendation.model.predict(recommendation.x_test)
    recommendation.precision_recall(recommendation.y_test, y_pred)
    recommendation.validate_model(True)
    recommendation.save_model(f'./model/{BUSINESS_UNIT}_model.pkl')

    # hyperparameters tuning 
    recommendation.grid_search()
    recommendation.parameter_combinations.to_csv(f'./data/{BUSINESS_UNIT}_grid_search_result.csv', index=False)
    recommendation.set_best_hyperparameter()   
    recommendation.save_model(f'./model/{BUSINESS_UNIT}_model.pkl')

    # feature importance
    recommendation.load_model(f'./model/{BUSINESS_UNIT}_model.pkl')
    plot_importance(recommendation.model)
    plt.savefig(f'./data/image/{BUSINESS_UNIT}_importance.png')

    # draw xgboost tree
    plot_tree(recommendation.model)
    fig = plt.gcf()
    fig.set_size_inches(150, 100)
    plt.savefig(f'./data/image/{BUSINESS_UNIT}_tree.png')

    # edm list predict
    # predict by open probability threshold
    recommend_list = recommendation.edm_recommend('threshold', max_recency=1000, min_open_prob=55)
    # predict by top N probability headcount
    recommend_list = recommendation.edm_recommend('headcount', max_recency=1000, headcount=500)
    recommend_list.to_excel(f'./data/edm_list/{BUSINESS_UNIT}_EDM_list.xlsx', index=False)