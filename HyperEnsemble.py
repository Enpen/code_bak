###主要是将hypergbm中的ensemble方法给单独抽离了出来，支持只用保存的文件去单独使用
import pandas as pd
import numpy as np
from hypernets.tabular.metrics import metric_to_scoring
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold
from hypergbm import HyperGBMEstimator
from hypernets.tabular.lifelong_learning import select_valid_oof
from hypernets.tabular.ensemble.voting import GreedyEnsemble
from hypernets.utils import logging
import warnings

warnings.filterwarnings("ignore")
logger = logging.get_logger(__name__)


class HyperEnsembel:
    def __init__(self, estimators, X_train, y_train, scorer, task, X_eval=None, y_eval=None, ensemble_size=20,
                 random_seed=623):
        self.scorer = scorer if scorer is not None else get_scorer('neg_log_loss')
        self.ensemble_size = ensemble_size
        self.task = task
        self.estimators = estimators[:self.ensemble_size]
        self.random_seed = random_seed
        self.ensemble_model = self.build_model(X_train, y_train, X_eval, y_eval)

    def build_model(self, X_train, y_train, X_eval, y_eval):
        ensemble = self.get_ensemble(self.estimators, X_train, y_train)
        if X_eval is None:
            logger.info('ensemble with oofs')
            oofs = self.get_oofs(X_train, y_train)
            assert oofs is not None
            if hasattr(oofs, 'shape'):
                y_, oofs_ = select_valid_oof(y_train, oofs)
                ensemble.fit(None, y_, oofs_)
            else:
                ensemble.fit(None, y_train, oofs)
        else:
            ensemble.fit(X_eval, y_eval)
        return ensemble

    def get_oofs(self, X, y):
        def get_oof(estimator, X, y, random_seed):
            iterators = StratifiedKFold(n_splits=len(estimator.cv_gbm_models_), shuffle=True, random_state=random_seed)
            # iterators = KFold(n_splits=len(estimator.cv_gbm_models_), shuffle=True,random_state=623)
            oof = np.zeros((len(X), 2))  ##这里的2是因为预测二分类的话，结果是两个类别分别的概率
            for n_fold, (train_idx, valid_idx) in enumerate(iterators.split(X, y)):
                x_val_fold, y_val_fold = X.iloc[valid_idx], y.iloc[valid_idx]
                oof[valid_idx] = estimator.cv_gbm_models_[n_fold].predict_proba(estimator.transform_data(x_val_fold))
            return oof

        oofs = None
        for i, estimator in enumerate(self.estimators):
            oof = get_oof(estimator, X, y, self.random_seed)
            if oofs is None:
                if len(oof.shape) == 1:
                    oofs = np.zeros((oof.shape[0], len(self.estimators)), dtype=np.float64)
                else:
                    oofs = np.zeros((oof.shape[0], len(self.estimators), oof.shape[-1]), dtype=np.float64)
            oofs[:, i] = oof
        return oofs

    def get_ensemble(self, estimators, X_train, y_train):
        return GreedyEnsemble(self.task, estimators, scoring=self.scorer, ensemble_size=self.ensemble_size)


train_data = pd.read_csv("./../resource/new_train.csv", index_col=0)
test_data = pd.read_csv("./../resource/new_test.csv", index_col=0)
history = pd.read_csv('./../resource/sep_total.csv')['model_path']

reward_metric = 'auc'

y = train_data.pop('claim')
X = train_data
estimators = [HyperGBMEstimator.load(modelfile) for modelfile in history]

ensemble_ = HyperEnsembel(estimators=estimators, X_train=X, y_train=y,
                          scorer=get_scorer(metric_to_scoring(reward_metric)), task='binary', random_seed=0,
                          ensemble_size=9)
ensemble_model = ensemble_.ensemble_model

pred_proba = ensemble_model.predict_proba(test_data)[:, 1].reshape(-1, 1)
_columns = ['id', 'claim']
_start_id = 957919
pred_proba_plus_id = np.hstack((np.arange(_start_id, len(pred_proba) + _start_id).reshape(-1, 1), pred_proba))
pf = pd.DataFrame(pred_proba_plus_id, columns=_columns)
pf['id'] = pf['id'].astype(int)
pf.to_csv("./../resource/hypergbm_out_e.csv", encoding='utf-8', index=None)
print('done successfully')






