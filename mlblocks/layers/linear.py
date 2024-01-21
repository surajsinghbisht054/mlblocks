

class LinearLayer:
    def __init__(self, neurons, label="Layer"):
        self.label = label
        self.neurons = neurons 
        self.ncount = len(self.neurons)
        self.wcount = 0
        if neurons:
            self.wcount = len(neurons[0].data)
        self.results = []
        
        
    def get_values(self):
        return {
            "label":self.label,
            "ncount":self.ncount,
            "wcount":self.wcount,
            "neurons":[neuron.get_values() for neuron in self.neurons]
        }
    
    def feed(self, arr, rcount=None, ccount=None):
        return [neuron.feed(arr) for neuron in self.neurons]
    
    def __repr__(self):
        return f"{self.label}({self.ncount}x{self.wcount})"
    
    def __len__(self):
        return self.ncount