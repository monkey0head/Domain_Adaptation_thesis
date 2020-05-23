from collections import defaultdict


class AvgLossLogger:
    """
    Class for averaging batch losses over epoch
    """
    def __init__(self):
        self.history = defaultdict(lambda: [0, 0])
     
    def store(self, prefix=None, **kwargs):
        for key, value in kwargs.items():
            if prefix is not None:
                name = "{}_{}".format(prefix, key)
            else:
                name = key
            self.history[name][0] += value
            self.history[name][1] += 1
        
    def reset_history(self):
        for name in self.history.keys():
            self.history[name] = [0, 0]
        
    def get_info(self):
        return {k: v[0]/v[1] for k, v in self.history.items() if v[1] != 0}
