from mlblocks.gradient.tensor import TensorValue
import torch

def test_gradient_calculation(W):
    a = W(-3.0)
    b = W(2.0)
    c = a + b
    d = (a * b) + (b**3)
    e = ((c ** 3)/0.4)+0.2
    d += (c * 2 + (b + a))
    d += (a * b).relu()**3
    d += (a * 2) + e.relu()
    g = (((d + b + e)/2)*0.5).exp()
    return (g, a, b)
g1, a1, b1 = test_gradient_calculation(lambda x: torch.tensor(x, requires_grad=True, dtype=torch.float32))
g1.backward()
g2, a2, b2 = test_gradient_calculation(TensorValue)
g2.backward()
comparision_float_precision = 6
test_pass = round(a1.grad.item(), comparision_float_precision) ==round(a2.grad, comparision_float_precision)
test_pass = test_pass and round(b1.grad.item(), comparision_float_precision)==round(b2.grad, comparision_float_precision)
# test if tensorvalue generating correct gradient value
assert test_pass is True