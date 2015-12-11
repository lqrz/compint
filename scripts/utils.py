class RunningAverage:
    def __init__(self):
        self.avg = 0.0
        self.count = 0

    def add_value(self, x):
        self.avg = (self.avg * self.count + float(x)) / (self.count + 1)
        self.count += 1

if __name__ == '__main__':
    ra = RunningAverage()
    l = [1, 5, 3, 7]
    for i in l:
        ra.add_value(i)
        print ra.avg