import torch
import torch.nn as nn

# first layer set up
w1 = torch.normal(0, 1, size=(3,4), requires_grad=True)
x  = torch.normal(0, 1, size=(1,4), requires_grad=True)
y  = x + 1
b1 = torch.normal(0, 1, size=(3,1), requires_grad=True)

# second layer set up
w2 = torch.normal(0, 1, size=(4,3), requires_grad=True)
b2 = torch.normal(0, 1, size=(4,1), requires_grad=True)
'''
Begin NN forward pass
'''
# q = w1.dot(x.T) + b1
q = torch.matmul(w1, torch.transpose(x, 0, 1)) + b1
q.retain_grad()
# h = ReLu(q)
m = nn.ReLU()
h = m(q)
h.retain_grad()
# p = w2.dot(h.T) + b2
p = torch.matmul(w2, h) + b2
p.retain_grad()
# compute loss
loss = torch.sum((p - torch.transpose(y, 0, 1)) * (p - torch.transpose(y, 0, 1)))
loss.backward()

# dL / dp
assert(torch.allclose(p.grad, 2*(p - torch.transpose(y,0,1)), atol=1e-3))
# dL / dw2
assert(torch.allclose(w2.grad, 2*torch.matmul((p -
    torch.transpose(y,0,1)),torch.transpose(h,0,1)), atol=1e-3))
# dL / db2
assert(torch.allclose(b2.grad, 2*(p - torch.transpose(y,0,1)), atol=1e-3))
# dL / dh
assert(torch.allclose(h.grad, 2*torch.matmul(torch.transpose(p,0,1)-y,
    w2).transpose(0,1), atol=1e-3))
# dL / dq
def relu_grad(a):
    a[a<0] = 0
    a[a>0] = 1
    return a

assert(torch.allclose(q.grad, h.grad * relu_grad(q),atol=1e-3))
# dL / dw1
assert(torch.allclose(w1.grad, torch.matmul(q.grad, x), atol=1e-3))
# dL / db1
assert(torch.allclose(b1.grad, q.grad, atol=1e-3))

