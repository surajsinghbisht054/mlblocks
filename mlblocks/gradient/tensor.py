import math

class TensorValue:
    def __init__(self, num, ops= None, left=None, right=None):
        self.num = float(num)
        self.ops = ops
        self.left = left
        self.right = right
        self.grad = 0.0
        self.leaf = True
        
        if self.left or self.right:
            self.leaf = False
    
    def __add__(self, y):
        return self.__class__(self.num + getattr(y, "num", y), ops="+", left=self, right=y)
    
    def __radd__(self,  y):
        return self.__add__(y)
    
    def __sub__(self, y):   
        return self.__add__(-y)
    
    def __rsub__(self,  y):
        return self.__add__(-y)
    
    def __mul__(self, y):     
        return self.__class__(self.num * getattr(y,"num", y), ops="*", left=self, right=y)
    
    def __rmul__(self,  y):
        return self.__mul__(y)
    
    def relu(self):
        return self.__class__(max(0, self.num), ops="rl", left=self)
    
    def exp(self):
        return self.__class__(math.exp(self.num), ops="exp", left=self)

    def __neg__(self):
        return self.__mul__(-1)
    
    def __truediv__(self, y):        
        return self.__mul__(y**-1)
    
    def __rtruediv__(self,  y):
        return self.__mul__(y**-1)
    
    def __pow__(self, y):
        return self.__class__(self.num ** getattr(y, "num", y), ops="pow", left=self, right=y)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.num})<{id(self)}>"
    
    
    def flush_gradient(self):
        if not self.leaf:
            self.grad = 0.0
        if isinstance(self.left, self.__class__):
            self.left.flush_gradient() 
        if isinstance(self.right, self.__class__):
            self.right.flush_gradient()
    
    def backward(self):
        self.grad = 1.0
        return self.calculate_gradient_backward()
        
    def calculate_gradient_backward(self,):
        r = getattr(self.right, "num", self.right)
        l = getattr(self.left, "num", self.left)
    
        # the derivative of f(x, y) = x + y with respect to x is simply 1
        if self.ops=="+":
            self.left.grad += (self.grad * 1.0)
            if isinstance(self.right, self.__class__):
                self.right.grad += (self.grad * 1.0)
                
        # the derivative of f(a, b) = a * b with respect to 'a' is 'b'.
        elif self.ops=="*":
            self.left.grad += (r * self.grad)
                
            if isinstance(self.right, self.__class__):
                self.right.grad += (l * self.grad)
        
        elif self.ops=="rl":
            self.left.grad += (int(self.num > 0) * self.grad)
        
        elif self.ops=="exp":
            self.left.grad += (self.num * self.grad)
                
        # the derivative of f(a, b) = a^b with respect to 'a' is 'b * a^(b-1)'
        elif self.ops=="pow":
            self.left.grad += ((r * (l ** (r - 1.0))) * self.grad)
                
            if isinstance(self.right, self.__class__):
                self.right.grad += ((l * (r ** (l - 1.0))) * self.grad)
        
    
        if isinstance(self.left, self.__class__):
            self.left.calculate_gradient_backward()
            self.left.flush_gradient()
            
        if isinstance(self.right, self.__class__):
            self.right.calculate_gradient_backward()
            self.right.flush_gradient()
