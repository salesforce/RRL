
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += float(val * n)
        self.count += n
        self.avg = round(self.sum / self.count,4)
        

class DAverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.values = {}

    def update(self, values):
        assert(isinstance(values, dict))
        for key, val in values.items():
            if not (key in self.values):
                self.values[key] = AverageMeter()
            self.values[key].update(val)
                   
    def average(self):
        average = {}
        for key, val in self.values.items():
            average[key] = val.avg                
        return average
        
    def __str__(self):
        ave_stats = self.average()
        return ave_stats.__str__()
