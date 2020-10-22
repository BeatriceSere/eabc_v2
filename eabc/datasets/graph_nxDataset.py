# -*- coding: utf-8 -*-

from eabc.datasets import Dataset
from eabc.data import Graph_nx

class graph_nxDataset(Dataset):
    
    def __init__(self, path, name, reader, transform = None, pre_transform = None):
        
        self.name = name
        self.reader = reader
        self.path = path
        self.transform = transform;
        self.pre_transform = pre_transform
        
        super(graph_nxDataset,self).__init__(self.path, self.transform, self.pre_transform)
        
    def process(self):
        examples,classes =self.reader(self.path)
        reader_out =zip(examples,classes)
        for x,y in reader_out:
            data = Graph_nx()
            data.x = x
            data.y = y
            self._data.append(data)
              

    def add_keyVal(self,idx,data):
        if isinstance(data,Graph_nx):
            self._data.append(data)
            self._indices.append(idx)
        else:
            raise ValueError("Invalid data inserted")
    
    def __repr__(self):
        return '{}{}()'.format(self.__class__.__name__, self.name.capitalize())