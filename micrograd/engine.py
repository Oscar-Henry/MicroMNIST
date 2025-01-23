import math

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children = (), _operation = ''):
        self.data = data
        self.grad = 0

        self._backward = lambda: None
        self._prev = set(_children)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data + other.data, (self, other))

        def _backward():
            self.grad += output.grad
            other.grad += output.grad
        output._backward = _backward

        return output
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad
        output._backward = _backward

        return output
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        output = Value(self.data**other, (self,))

        def _backward():
            self.grad = (other * self.data**(other - 1)) * output.grad
        output._backward = _backward

        return output
    
    def __exp__(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
    
        def _backward():
            self.grad += out.data * out.grad # NOTE: in the video I incorrectly used = instead of +=. Fixed here.
        out._backward = _backward
    
        return out
    
    def relu(self):
        output = Value(0 if self.data < 0 else self.data, (self,), 'ReLu')

        def _backward():
            self.grad += (output.data > 0) * output.grad
        output._backward = _backward

        return output
    
    def softmax(self, values):
        S_i = math.exp(self.data) / sum(math.exp(val.data) for val in values),
        output = Value(S_i, ((self, ) + values))

        def _backward():
            self.grad += S_i * (1 - S_i)
            x = sum(math.exp(val.data) for val in values)
            for val in values:
                S_k = math.exp(val.data) / x
                val.grad -= S_i * S_k
            output._backward = _backward()

        return output

    def backProp(self, error, learning_rate):
        return error
    
    def __neg__(self): # -self
        return self * -1
    
    def __radd__(self, other): # other + self
        return self + other
    
    def __sub__(self, other): # self - other
        return self + (-other)
    
    def __rsub__(self, other): # other - self
        return other + (-self)
    
    def __rmul__(self, other): # other * self
        return self * other
    
    def __truediv__(self, other): # self / other
        return self * other**-1
    
    def __rtruediv__(self, other): # other / self
        return other * self**-1
    
    def backward(self):

        topological_sort = []
        visited = set()
        
        # Topological sort of all childrens from root
        def build(root):
            if root not in visited:
                visited.add(root)
                for child in root._prev:
                    build(child)
                topological_sort.append(root)
        
        build(self)

        # Apply chain rule to get gradients
        self.grad = 1
        for node in reversed(topological_sort):
            node._backward
    
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
