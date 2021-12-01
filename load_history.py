#### 在载入history的时候支持最优单模型+不同的随机种子，最优单模型+不同的随机种子+不同的数据预处理方法

    def load_history(space_fn, filepath, load_mode = 0):
        if load_mode == 0:
            with open(filepath, 'r') as input:
                line = input.readline()
                history = TrialHistory(line.strip())
                while line is not None and line != '':
                    line = input.readline()
                    if line.strip() == '':
                        continue
                    fields = line.split('|')
                    assert len(fields) >= 4, f'Trial format is not correct. \r\nline:[{line}]'
                    sample = space_fn()
                    vector = [float(n) if n.__contains__('.') else int(n) for n in
                              fields[1].replace('[', '').replace(']', '').split(',')]
                    sample.assign_by_vectors(vector)
                    if len(fields) > 4:
                        model_file = fields[4]
                    else:
                        model_file = None
                    trial = Trial(space_sample=sample, trial_no=int(fields[0]), reward=float(fields[2]),
                                  elapsed=float(fields[3]), model_file=model_file)
                    history.append(trial)
                return history
        else:
            best_trial = None
            best_vectors = None
            best_reward = None
            history = None
            with open(filepath, 'r') as input:
                line = input.readline()
                history = TrialHistory(line.strip())
                while line is not None and line != '':
                    line = input.readline()
                    if line.strip() == '':
                        continue
                    fields = line.split('|')
                    assert len(fields) >= 4, f'Trial format is not correct. \r\nline:[{line}]'
                    if best_reward == None:
                        best_reward = float(fields[2])
                        continue
                    if history.optimize_direction == 'min':
                        if float(fields[2]) < best_reward:
                            best_reward = float(fields[2])
                            best_trial = fields
                            best_vectors = [float(n) if n.__contains__('.') else int(n) for n in
                                           fields[1].replace('[', '').replace(']', '').split(',')]
                    else:
                        if float(fields[2]) > best_reward:
                            best_reward = float(fields[2])
                            best_trial = fields
                            best_vectors = [float(n) if n.__contains__('.') else int(n) for n in
                                           fields[1].replace('[', '').replace(']', '').split(',')]
            if load_mode == 1:
                for i in range(6):
                    sample = space_fn()
                    sample.assign_by_vectors(best_vectors)
                    trial = Trial(space_sample=sample, trial_no=i+1, reward=float(fields[2]),
                                  elapsed=float(fields[3]), model_file=None)
                    history.append(trial)
                return history
            elif load_mode == 2:
                sample_ = space_fn()
                for index, param in enumerate(sample_.params_iterator):
                    if 'scaler' in param.alias:
                        if best_vectors[index]:  ##有scaler操作
                            for i in range(5):
                               best_vectors[-1] = i
                               sample = space_fn()
                               sample.assign_by_vectors(best_vectors)
                               trial = Trial(space_sample=sample, trial_no=i+1, reward=float(fields[2]),
                                             elapsed=float(fields[3]), model_file=None)
                               history.append(trial)
                            sample = space_fn()
                            best_vectors[index] = 0
                            best_vectors.pop()
                            sample.assign_by_vectors(best_vectors)
                            trial = Trial(space_sample=sample, trial_no=i + 1, reward=float(fields[2]),
                                          elapsed=float(fields[3]), model_file=None)
                            history.append(trial)
                            break
                        else:  ##原本没有scaler操作
                            sample = space_fn()
                            sample.assign_by_vectors(best_vectors)
                            trial = Trial(space_sample=sample, trial_no=1, reward=float(fields[2]),
                                          elapsed=float(fields[3]), model_file=None)
                            history.append(trial)
                            best_vectors[index] = 1 ### scaler置为True
                            best_vectors.append(0) ##扩充长度
                            for i in range(5):  ##第一个就用默认的
                                best_vectors[-1] = i
                                sample = space_fn()
                                sample.assign_by_vectors(best_vectors)
                                trial = Trial(space_sample=sample, trial_no=i + 2, reward=float(fields[2]),
                                              elapsed=float(fields[3]), model_file=None)
                                history.append(trial)
                            break
                return history
            else:
                assert 1==2,'invalid load_mode, please check it again'
