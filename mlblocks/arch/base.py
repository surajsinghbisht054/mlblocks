import json
from abc import ABC, abstractmethod


class AbstractBaseNet(ABC):
    layers = []
    pre_feed_hook = None
    post_feed_hook = None
                
    def __repr__(self):
        o = ['input(*)']
        o += [repr(i) for i in self.layers]
        return ' -> '.join(o)
        
    def feed(self, arr):
        if self.pre_feed_hook:
            arr = self.pre_feed_hook(arr)
        
        for layer in self.layers:
            arr = layer.feed(arr)
        
        if self.post_feed_hook:
            arr = self.post_feed_hook(arr)
        return arr
    
    def get_parameters(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                for weight in neuron.data:
                    yield weight
                yield neuron.bias
    
    def softmax(self, value_array):
        exp_logits = [val.exp() for val in value_array]
        sum_exp_logits = sum(exp_logits)
        softmax_probs = [exp_logit / sum_exp_logits for exp_logit in exp_logits]
        return softmax_probs
    
    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_loss(self, *args, **kwargs):
        pass
    
    def mean_square_error(self, predicted_arr, target_arr):
        return sum([(x-y)**2 for x,y in zip(predicted_arr, target_arr)])/len(target_arr)
    
    def save(self, filename):
        
        layers_data = [d.get_values() for d in self.layers]
        with open(filename, "w") as fp:
            json.dump(layers_data, fp)

    
    def load(self, filename, layerKlass, Neuronklass, TensorValueKlass):
        with open(filename, "r") as fp:
            dump = json.load(fp)
            self.layers = []
            for layer in dump:
                l = layerKlass([])
                l.label = layer['label']
                l.ncount = layer['ncount']
                l.wcount = layer['wcount']
                l.neurons = []
                for ndata in layer['neurons']:
                    o = Neuronklass([])
                    o.data = [TensorValueKlass(i) for i in ndata['data']]
                    o.bias = TensorValueKlass(ndata['bias'])
                    o.wcount = ndata['wcount']
                    l.neurons.append(o)

                self.layers.append(l)
