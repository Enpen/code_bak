from hypergbm.search_space import GeneralSearchSpaceGenerator
from hypergbm.estimators import XGBoostEstimator,LightGBMEstimator,CatBoostEstimator
from hypernets.core.search_space import Int, Choice, Real, Bool
from hypergbm.cfg import HyperGBMCfg as cfg
from hypergbm.estimators import LightGBMEstimator, XGBoostEstimator, CatBoostEstimator, HistGBEstimator
from hypergbm.pipeline import DataFrameMapper
from hypergbm.sklearn.sklearn_ops import numeric_pipeline_simple, numeric_pipeline_complex, \
    categorical_pipeline_simple, categorical_pipeline_complex, \
    datetime_pipeline_simple, text_pipeline_simple
from hypernets.core import randint
from hypernets.core.ops import ModuleChoice, HyperInput
from hypernets.core.search_space import HyperSpace, Choice, Int
from hypernets.tabular.column_selector import column_object
from hypernets.utils import logging, get_params

class MyGeneralSearchSpaceGenerator(GeneralSearchSpaceGenerator):
    def __init__(self,**kwargs,):
        super(MyGeneralSearchSpaceGenerator, self).__init__(**kwargs)
    def create_preprocessor(self, hyper_input, options):
        cat_pipeline_mode = options.pop('cat_pipeline_mode', cfg.category_pipeline_mode)
        num_pipeline_mode = options.pop('num_pipeline_mode', cfg.numeric_pipeline_mode)
        dataframe_mapper_default = options.pop('dataframe_mapper_default', False)
        pipelines = []
         
        # text
        if cfg.text_pipeline_enabled:
            pipelines.append(text_pipeline_simple()(hyper_input))

        # category
        if cfg.category_pipeline_enabled:
            if cat_pipeline_mode == 'simple':
                pipelines.append(categorical_pipeline_simple()(hyper_input))
            else:
                pipelines.append(categorical_pipeline_complex()(hyper_input))

        # datetime
        if cfg.datetime_pipeline_enabled:
            pipelines.append(datetime_pipeline_simple()(hyper_input))

        # numeric
        if num_pipeline_mode == 'customize':
            from hypergbm.pipeline import Pipeline
            import numpy as np
            from hypergbm.sklearn.transformers import SimpleImputer
            from hypernets.tabular import column_selector
            def numeric_pipeline_customize(impute_strategy='mean', seq_no=0):
                pipeline = Pipeline([
                    SimpleImputer(missing_values=np.nan, strategy=impute_strategy,
                                  name=f'numeric_imputer_{seq_no}', force_output_as_float=True),
                ],
                    columns=column_selector.column_number_exclude_timedelta,
                    name=f'numeric_pipeline_simple_{seq_no}',
                )
                return pipeline
            pipelines.append(numeric_pipeline_customize()(hyper_input))
        elif num_pipeline_mode == 'simple':
            pipelines.append(numeric_pipeline_simple()(hyper_input))
        elif num_pipeline_mode == 'skip':
            return None
        else:
            pipelines.append(numeric_pipeline_complex()(hyper_input))

        preprocessor = DataFrameMapper(default=dataframe_mapper_default, input_df=True, df_out=True,
                                       df_out_dtype_transforms=[(column_object, 'int')])(pipelines)

        return preprocessor

    @property
    def default_xgb_init_kwargs(self):
        _result_kwargs=super().default_xgb_init_kwargs
        xgb_init_kwargs={
                         'objective': 'binary:logistic',
                         'tree_method': 'gpu_hist',
                        #  'gpu_id': '0',
                        #  'predictor': 'gpu_predictor', 
                         'n_jobs':-1,
                        #  'booster':Choice(['gbtree','dart']),
                         'booster':'gbtree',
                         'colsample_bytree': Real(0.3,1,step=0.1),
                         'colsample_bylevel':Real(0.3,1,step=0.1),
                         ##xgboost训练太慢了，所以这里极限不要超过1w
                        #  'n_estimators':Choice([200,400,800,1200,1400,1600,1800,2000,2300,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000,12000,14000,16000,18000,20000]),
                         'n_estimators':Choice([200,400,800,1200,1400,1600,1800,2000,2300,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000,12000,14000]),
                         'gamma': Choice([0.5, 1, 1.5, 2, 3, 4, 5]),
                         'reg_alpha': Choice([0,0.1,0.2,0.3, 0.5,0.7,1,2,5,7,10,13,15,20,40,60,80,100]),
                         'reg_lambda': Choice([0,0.001,0.005, 0.01, 0.05,0.1,0.5,0.8,1]),
                        #  'min_child_weight':Choice([1,2,3,5,7,10]),
                         'min_child_weight':Int(1,200,4),
                         'subsample': Real(0.3, 1, step=0.1),
                         'max_depth':  Choice([2,3,4,5,6,7,8,9]),
                         'learning_rate':Choice([0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6]),
                         'eval_metric': 'auc',
                         }
        # xgb_init_kwargs = {
        #     'n_jobs': -1,
        #     'tree_method': 'gpu_hist',
        #     'gpu_id': '0',
        #     'objective': 'reg:squarederror',
        #     'booster': 'gbtree',
        #     'colsample_bytree ': 0.6,
        #     'colsample_bylevel': 1,
        #     'n_estimators': 10000,
        #     'gamma': None,
        #     'reg_alpha': 20,
        #     'reg_lambda': 9,
        #     'min_child_weight': 256,
        #     'subsample': 0.8,
        #     'max_depth': 11,
        #     'learning_rate':Choice([0.002,0.003,0.004,0.005,0.006,0.007]),
        #     'eval_metric': 'RMSE',
        #     'importance_type': 'total_gain',
        # }
        for k,v in xgb_init_kwargs.items():
            _result_kwargs.update({k:v})
        return _result_kwargs

    @property
    def default_lightgbm_init_kwargs(self):
        _result_kwargs = super().default_lightgbm_init_kwargs
        lightgbm_init_kwargs = {'colsample_bytree': Real(0.3, 1, step=0.1),
                                # 'n_estimators': Choice([200,400,800,1200,1600,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000,12000,14000,16000,18000,20000]),
                                'n_estimators': Choice([200,400,800,1200,1600,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000,12000,14000]),
                                'boosting_type': Choice(['gbdt', 'goss']),
                                'learning_rate': Choice([0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6]),
                                'max_depth':  Choice([2,3,4,5,6,7,8,9]),
                                'num_leaves': Int(5, 500, 5),
                                'reg_alpha': Choice([0,0.1,0.2,0.3, 0.5,0.7,1,2,5,7,10,13,15,20,40,60,80]),
                                'reg_lambda': Choice([0,0.001,0.005,0.01,0.03,0.05,0.1,0.3,0.5,0.8,1]),
                                'subsample': Real(0.3, 1, step=0.1),
                                'min_child_samples': Int(2,100,step=1),
                                # 'min_child_weight': Choice([0.001, 0.002]),
                                'min_child_weight': Int(1,300,6),
                                # 'bagging_fraction': Real(0.5, 1, step=0.1),
                                'metric': 'auc',
                                'n_jobs':48,
                                # 'verbose':1,
                                }
        # lightgbm_init_kwargs = {
        #                         'learning_rate': 0.1,
        #                         'objective': 'binary',
        #                         'boosting_type': 'gbdt',
        #                         'num_leaves': 6,
        #                         'max_depth': 2,
        #                         'n_estimators': 40000,
        #                         'reg_alpha': 25.0,
        #                         'reg_lambda': 76.7,
        #                         'random_state': 0,
        #                         'bagging_seed': 0, 
        #                         'feature_fraction_seed': 0,
        #                         'n_jobs': -1,
        #                         'subsample': 0.98,
        #                         'subsample_freq': 1,
        #                         'colsample_bytree': 0.69,
        #                         'min_child_samples': 54,
        #                         'min_child_weight': 256,
        #                         'metric': 'AUC',
        #                         'verbosity': -1,
        #                         }
        for k, v in lightgbm_init_kwargs.items():
            _result_kwargs.update({k: v})
        return _result_kwargs

    @property
    def default_catboost_init_kwargs(self):
        _result_kwargs = super().default_catboost_init_kwargs
        catboost_init_kwargs={
                            #   'objective': 'binary:logistic',
                              'task_type': 'GPU',
                            #   'devices': '0',
                            #   'gpu_ram_part':0.6,
                            #   'n_estimators':Choice([200,400,800,1200,1600,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000,12000,14000,16000,18000,20000]),
                              'n_estimators':Choice([200,400,800,1200,1600,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000,12000,14000]),
                              'depth': Choice([2,3,4,5,6,7,8,9]),
                              'learning_rate': Choice([0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6]),
                              'l2_leaf_reg':Choice([0,1,2,3,5,7,10,13,15,20,30,40,50,60,70,80,90,100]),
                              'min_data_in_leaf': Int(5,100,step=1),
                              'leaf_estimation_method':Choice(['Newton','Gradient']),
                              # 'subsample': Real(0.1, 1, step=0.1), ##closed Cause Bayesian
                              'bootstrap_type':Choice(['Poisson','Bayesian','Bernoulli']),
                            #   'bootstrap_type':Choice(['Bayesian','Bernoulli']),
                            #   'loss_function':'RMSE',
                              'eval_metric': 'AUC',
                              }
        # catboost_init_kwargs = {
        #     'task_type': 'GPU',
        #     'devices': '0',
        #     'n_estimators': Choice(
        #         [10000,10000]),
        #     # 'n_estimators':Choice([200,400,800,1200,1400,1600,1800,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000]),
        #     'depth': Choice([8,8]),
        #     'learning_rate': Choice(
        #         [0.002,0.003,0.004,0.005,0.006,0.007]),
        #     'l2_leaf_reg': Choice([0.01,0.01]),
        #     'min_data_in_leaf': Choice([64,64,]),
        #     'leaf_estimation_method': 'Gradient',
        #     'subsample': 0.8, ##closed Cause Bayesian
        #     'bootstrap_type': 'Poisson',
        #     'max_bin':280,
        #     'loss_function':'RMSE',
        #     'eval_metric': 'RMSE',
        # }
        for k, v in catboost_init_kwargs.items():
            _result_kwargs.update({k: v})
        return _result_kwargs
    
    @property
    def default_lightgbm_fit_kwargs(self):
        return {'early_stopping_rounds':100}

    @property
    def default_xgb_fit_kwargs(self):
        return {'early_stopping_rounds':100}
    
    @property
    def default_catboost_fit_kwargs(self):
        return {'early_stopping_rounds':100}


    @property
    def estimators(self):
        r = {}
        if self.enable_lightgbm:
            r['lightgbm'] = (LightGBMEstimator, self.default_lightgbm_init_kwargs, self.default_lightgbm_fit_kwargs)
        if self.enable_xgb:
            r['xgb'] = (XGBoostEstimator, self.default_xgb_init_kwargs, self.default_xgb_fit_kwargs)
        if self.enable_catboost:
            r['catboost'] = (CatBoostEstimator, self.default_catboost_init_kwargs, self.default_catboost_fit_kwargs)
        if self.enable_histgb:
            r['histgb'] = (HistGBEstimator, self.default_histgb_init_kwargs, self.default_histgb_fit_kwargs)
        return r
