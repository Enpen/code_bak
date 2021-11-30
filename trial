class PlaybackSearcher(Searcher):
    def __init__(self, history: TrialHistory, top_n=None, reverse=False,
                 optimize_direction=OptimizeDirection.Minimize,playback_mode = 0):
        assert history is not None
        assert len(history.trials) > 0
        self.history = history
        if playback_mode == 1:
            from hypernets.core import randint
            from copy import deepcopy
            _history = self.history.get_best()
            self.history.trials = []
            for i in range(6):
                self.history.trials.append(deepcopy(_history))
            for i,trail in enumerate(self.history.trials):
                for param in trail.space_sample.hyper_params:
                    if '.random_state' in str(param.alias):
                        param._assigned = False
                        param.assign(randint())
                        print(param.value)
                        break

        elif playback_mode == 2:
            from hypernets.core import randint
            from copy import deepcopy
            _history = self.history.get_best()
            _vectors = _history.space_sample.vectors
            self.history.trials = []
            for i in range(6):
                self.history.trials.append(deepcopy(_history))
            for param, index in zip(_history.space_sample.assigned_params_stack, _vectors):
                if 'scaler' in param.alias:
                    if param.value:  ##有scaler操作
                        for i in range(5):
                            a = self.history.trials[i].space_sample.vectors
                            a[-1] = i
                            self.history.trials[i].space_sample._vectors = a  ##分别是5种scaler的方法
                        self.history.trials[i].space_sample.vectors[index] = 0  ## scaler置为False
                        self.history.trials[i].space_sample.vectors.pop()
                    else:  ##原本没有scaler操作
                        for i in range(5):  ##第一个就用默认的
                            a = self.history.trials[i].space_sample.vectors
                            a[-1] = i
                            self.history.trials[i].space_sample.vectors[index] = 1  ## scaler置为True
                            self.history.trials[i].space_sample.vectors.append(i)
                    break
            for i,trail in enumerate(self.history.trials):
                for param in trail.space_sample.hyper_params:
                    if '.random_state' in str(param.alias):
                        param._assigned = False
                        param.assign(randint())
                        print(param.value)
                        break
                        
                        ##本来想的是从这里做一个用最优模型做6个随机数不同的操作，然后最优模型+6个随机种子+6中预处理方法，结果你会发现，第二个功能的vectors根本就不可以改，尤其是原来没有scaler操作，所以就
                        没有最后的选择scaler的assigned_params_stack记录，所以还是只能从load_history入手
                        
         
