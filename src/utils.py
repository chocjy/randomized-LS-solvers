import numpy as np
import cPickle as pickle
import json

def add(x, y):
    x += y
    return x

def unifSampling(row, n, s):
    #row = row.getA1()
    p = s/n
    coin = np.random.rand()
    if coin < p:
        return row/p

def pickle_load(filename):
    return pickle.load(open( filename, 'rb' ))

def pickle_write(filename,data):
    with open(filename, 'w') as outfile:
        pickle.dump(data, outfile, True)

def json_write(filename,*args):
    with open(filename, 'w') as outfile:
        for data in args:
            json.dump(data, outfile)
            outfile.write('\n')

class BlockMapper:
    """
    process data after receiving a block of records
    """
    def __init__(self, blk_sz=5e4):
        self.blk_sz = blk_sz
        self.keys = []
        self.data = []
        self.sz = 0

    def __call__(self, records, **kwargs):
        for r in records:
            self.keys.append(r[0])
            self.data.append(r[1])
            self.sz += 1
                
            if self.sz >= self.blk_sz:
                #for key, value in self.process(**kwargs):
                #    yield key, value
                for result in self.process(**kwargs):
                    yield result
                self.keys = []
                self.data = []
                self.sz = 0

        if self.sz > 0:
            #for key, value in self.process(**kwargs):
                #yield key, value
            for result in self.process(**kwargs):
                yield result

        #for key, value in self.close():
        #    yield key, value
        for result in self.close():
            yield result

    def process(self,**kwargs):
        return iter([])
    
    def close(self):
        return iter([])

