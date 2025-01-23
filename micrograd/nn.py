import random
from micrograd.engine import Value
from micrograd.engine import softmax

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, act):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.act = act

    def __call__(self, x):
        activation = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)

        if self.act == "Lin":
            return activation
        elif self.act == "ReLu":
            return activation.relu()
        
    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{self.act} Neuron ({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, act):
        self.neurons = [Neuron(nin, act) for _ in range(nout)]
        self.act = act

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        
        if self.act != "Softmax":
            return out[0] if len(out) == 1 else out
        else:
            return [outi.softmax(out) for outi in out]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts, act):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], act[i]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"