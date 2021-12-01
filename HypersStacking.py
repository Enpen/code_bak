### 这里主要是将hypergbm的stacking方法给单独独立了出来，支持直接使用模型保存路径的list进行操作。
### 同时支持新特征的生成(多层stacking)

from logging import FATAL
import os
import pickle
from hypernets.core import random_state
import pandas as pd
import numpy as np
import hashlib
from scipy.stats import kendalltau
from sklearn.model_selection import train_test_split
from hypergbm import make_experiment
from hypernets.searchers import PlaybackSearcher
from hypernets.utils.param_tuning import search_params
from hypernets.core.search_space import Int,Choice,Real,Bool
from hypernets.core.trial import TrialHistory
from hypergbm.search_space import search_space_general
from sklearn.model_selection import train_test_split
from hypernets.searchers import EvolutionSearcher, RandomSearcher
from hypergbm.search_space import GeneralSearchSpaceGenerator
from hypernets.tabular.metrics import metric_to_scoring
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold
import joblib
# from Mtools import *
from hypernets.core import randint
import time
import random
from itertools import combinations
import lightgbm as lgb
import catboost
from sklearn.linear_model import LogisticRegression,LinearRegression
from MyGeneralSearchSpaceGenerator import MyGeneralSearchSpaceGenerator
from hypergbm import HyperGBMEstimator
from hypernets.tabular.ensemble.voting import HyperStacking
import warnings
warnings.filterwarnings("ignore")
class MattHyperStacking(HyperStacking):
    def __init__(self, **kwargs):
        super(MattHyperStacking, self).__init__(**kwargs)
    def generate_new_features_over(self, df):
        combs = list(combinations(df.columns, 2))
        for i, j in combs:
            column_name = '%s-%s' % (i, j)
            df[column_name] = df[i] - df[j]
            column_name = '%s+%s' % (i, j)
            df[column_name] = df[i] + df[j]
            column_name = '%s*%s' % (i, j)
            df[column_name] = df[i] * df[j]
            column_name = '%s/%s' % (i, j)
            df[column_name] = df[i] / df[j]
        return df

    def get_topN_predictions(self,X,y=None):
        modelsPipeline=[self._estimators[i] for i in [0,1,3]]
        test_mode=False
        if y is None:
            test_mode=True
        result = np.zeros((X.shape[0],(len(modelsPipeline)*self.predict_len_+self.predict_len_)))
        if test_mode:
            for i,model in enumerate(modelsPipeline):
                result[:,i]=model.predict(X)
            result[:,len(modelsPipeline)]=self.predict_proba(X).reshape(X.shape[0])
            if result[0][-1]>10:
                result[:,len(modelsPipeline)]=result[:,len(modelsPipeline)] / 2
        else:
            for i, fold in enumerate(self.kfold(X, y,self.cv_nfolds, stratify=self.stratify, seed=self.random_state, shuffle=True)):
                X_train, y_train, X_test, y_test, train_index, test_index = fold
                for j, model in enumerate(modelsPipeline):
                    X_test_ = model.transform_data(X_test)
                    result[test_index,j] = model.cv_gbm_models_[i].predict(X_test_)
            next_layer_train, y = self.stack(X,y,self.modelsPipeline,stacking_k=self.cv_nfolds,stratify=self.stratify,seed=self.random_state,add_diff=self.add_diff,need_fit=False,layers_name='layer1')
            for i, fold in enumerate(self.kfold(next_layer_train, y, 3, stratify=self.stratify, seed=self.random_state, shuffle=True)):
                X_train, y_train, X_test, y_test, train_index, test_index = fold
                for j in range(2):
                    result[test_index,3] +=self.fin_models_cv[i*2+j].predict(X_test)
            result[:,len(modelsPipeline)]=result[:,len(modelsPipeline)]/2                
        result = pd.DataFrame(result,columns=['Top1','Top2','Top3','TopS'])
        result = self.generate_new_features_over(result)
        _columns = list(X.columns)+list(result.columns)
        if test_mode:
            result = pd.DataFrame(np.hstack((X,result)),columns=_columns)
        else:
            result = pd.DataFrame(np.hstack((X,result,np.array(y).reshape(-1,1))),columns=_columns+['loss'])
        return result


train_data =pd.read_csv("./../resource/new_train.csv",index_col=0)
test_data = pd.read_csv("./../resource/new_test.csv",index_col=0)
history = pd.read_csv('./../resource/sep_total.csv')['model_path']

reward_metric = 'auc'

y = train_data.pop('claim')
X = train_data


estimators = [HyperGBMEstimator.load(modelfile)  for modelfile in  history]


mattHeamyStacking = MattHyperStacking(task='binary',estimators=estimators,X_train=X,y_train=y,random_state=0,reward_score=get_scorer(metric_to_scoring(reward_metric)),
                                      stacking_mode='complex',stacking_trials=10,
                                      full_test=False,need_data_transform=True,add_old_features=False)

pred_proba = mattHeamyStacking.predict_proba(test_data)[:,1].reshape(-1,1)
# print(mattHeamyStacking.best_score_)
_columns = ['id','claim']
_start_id = 957919
pred_proba_plus_id = np.hstack((np.arange(_start_id,len(pred_proba)+_start_id).reshape(-1,1),pred_proba))
pf = pd.DataFrame(pred_proba_plus_id,columns=_columns)
pf['id'] = pf['id'].astype(int)
pf.to_csv("./../resource/hypergbm_out_h.csv",encoding='utf-8',index=None)
print('done successfully')



