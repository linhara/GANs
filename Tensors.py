import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32).to(DEVICE)

print(my_tensor)
x = torch.eye(5)
print(x)
print(torch.arange(5,19,3))
print(torch.linspace(0.1,0.63,7))
print(torch.empty((1,5)).normal_(0,1))
print(torch.empty((1,5)).uniform_(-1,1))
print(my_tensor.int())

import numpy as np
npar = np.array([1,2,3])
tarr = torch.from_numpy(npar)
npar2 = tarr.numpy()

x = torch.tensor([1,2,3])
y = torch.tensor([4,5,6])

print(x/y)

print(x.add_(2))

x1 = torch.rand((2,5))
x2 = torch.rand((5,3))

x3 = torch.mm(x1,x2)
x3 = x1.mm(x2)

batch = 32
n = 10
m = 20
p = 30

t1 = torch.rand((batch,n,m))
t2 = torch.rand((batch,m,p))

out_bmm = torch.bmm(t1,t2) #(batch, n, p)

'''BROADCASTING LIKE NUMPY'''

sum_x = torch.sum(x, dim=0)
values, i = torch.max(x,dim=0)
values, i = torch.min(x,dim=0)
ii = torch.argmax(x,dim=0)
torch.mean(x.float(),dim = 0)
sorted, i = torch.sort(y, dim=0, descending=False)

z = torch.clamp(x,min=0,max=4)
print(z)
tru = torch.tensor([1,2,3])
fls = torch.tensor([1,0,3])

print(torch.any(tru))
print(torch.all(tru))
print(torch.any(fls))
print(torch.all(fls))

bs = 10
features = 25
x = torch.rand((bs,features))

print(x[0].shape)
print(x[:,0].shape)

x=torch.arange(10)
i = [1,4,7]
print(x[i])

m = torch.rand((3,5))
rows = torch.tensor([1,0])
cols = torch.tensor([4,0])
print(m[rows,cols].shape)

x = torch.arange(10)
print(x[(x<4)&(x>0)])
print(x[x.remainder(2)==2])
print(torch.where(x>5, x, x*2))

print(m.ndimension())
print(m.numel())


x=torch.arange(9)

x_3x3 = x.view(3,3)
x_3x3 = x.reshape(3,3)

print(x_3x3.view(9))
y = x_3x3.t()
try:
    print(y.view(9))
except:
    print("lmao not continous, cant view must 'reshape'")
print(y.reshape(9))
print(y.contiguous().view(9))

x1 = torch.rand((2,5))
x2 = torch.rand((2,5))
print(torch.cat((x1,x2), dim=0).shape)
print(torch.cat((x1,x2), dim=1).shape)

z = x1.view(-1)
print(z.shape)

batch = 64
x = torch.rand((batch,2,5))
z = x.view(batch,-1)
print(z.shape)

print(x.shape)
z = x.permute(0,2,1)
print(z.shape)

x = torch.arange(10)
print(x.unsqueeze(0).shape)
print(x.squeeze(0).shape)

print(x.unsqueeze(1).shape)
'''INTENDED ERROR'''
print(x.squeeze(1).shape)