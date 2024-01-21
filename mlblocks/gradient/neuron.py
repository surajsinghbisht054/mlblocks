from .tensor import TensorValue

class Neuron:
    def __init__(self, weights, bias=1, grad=0):
        self.data = [TensorValue(i) for i in weights]
        self.bias = TensorValue(bias)
        self.wcount = len(self.data)
        self.grad = grad
    
    def get_values(self):
        return {"data":[i.num for i in self.data], "bias":self.bias.num, "wcount":self.wcount}
    
    def activation(self, val):
        return val.relu()
    
    def feed(self, arr):
        return self.activation(sum([weight*num for weight, num in zip(self.data, arr)]) + self.bias)
        
    def __repr__(self):
        return f"N({self.wcount}xW.)"
    
    def __iter__(self):
        return iter(self.data)
    
    def __len__(self):
        return len(self.data)