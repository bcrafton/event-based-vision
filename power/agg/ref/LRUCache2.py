
from collections import OrderedDict 
  
class LRUCache: 
  
    def __init__(self, name, capacity):
        self.name = name 
        self.cache1 = OrderedDict()
        self.cache2 = OrderedDict() 
        self.capacity = capacity 

    def contains(self, key): 
        size = len(self.cache1) + len(self.cache2)
        assert (size <= self.capacity)

        ret = (key in self.cache1) or (key in self.cache2)
        return ret
  
    def put(self, key: int, value: int) -> None:
        assert (not self.contains(key))
    
        if value == 0:
            assert len(self.cache1) < self.capacity, self.name
            self.cache1[key] = value
        else: 
            self.cache2[key] = value
            self.cache2.move_to_end(key)

        total = len(self.cache1) + len(self.cache2)
        if total > self.capacity:
            (key, value) = self.cache2.popitem(last = False)
            assert (value > 0)
            return (key, value)
        else:
            return None

    def read(self, key):
        assert (self.contains(key))
        if key in self.cache2:
            assert (key not in self.cache1)
            value = self.cache2.pop(key)
            self.cache1[key] = value

    def access(self, key):
        assert (self.contains(key))

        if key in self.cache2:
            self.cache2.move_to_end(key)
            self.cache2[key] += 1
        
        elif key in self.cache1:
            value = self.cache1.pop(key)
            self.cache2[key] = value + 1
            self.cache2.move_to_end(key)
        
        

