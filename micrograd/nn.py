import random
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, soft = False):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.soft = soft

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)

        if self.soft:
            return act
        else:
            return act.relu()
        
    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'Softmax' if self.soft else 'ReLu'} Neuron ({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, soft):
        self.neurons = [Neuron(nin, soft) for _ in range(nout)]
        self.soft = soft

    def __call__(self, x):
        out = [n(x) for n in self.neurons]

        if self.soft:
            return [outi.softmax(out) for outi in out]
        else:
            return out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], soft = i == len(nouts) - 1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"