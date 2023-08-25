import torch
import numpy as np
import os
import GPUtil
import time

import torch.quantization

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = 'cuda'
torch.set_default_device(device)


def nn_autograd_simple(model, points, order, axis=0):
    points.requires_grad = True
    f = model(points).sum()
    for i in range(order):
        grads, = torch.autograd.grad(f, points, create_graph=True)
        f = grads[:, axis].sum()
    return grads[:, axis]


def grid16():
    x = torch.from_numpy(np.linspace(0, 1, 101)).half()
    t = torch.from_numpy(np.linspace(0, 1, 101)).half()

    coord_list = []
    coord_list.append(x)
    coord_list.append(t)

    grid = torch.cartesian_prod(x, t).half().to(device)
    return x, t, grid


def grid32():
    x = torch.from_numpy(np.linspace(0, 1, 101)).float()
    t = torch.from_numpy(np.linspace(0, 1, 101)).float()

    coord_list = []
    coord_list.append(x)
    coord_list.append(t)

    grid = torch.cartesian_prod(x, t).float().to(device)
    return x, t, grid


def bconds_float32(x, t, grid):
    # Initial conditions at t=0
    bnd1 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float32))).float().to(device)

    # u(0,x)=sin(pi*x)
    bndval1 = torch.sin(torch.pi * bnd1[:, 0]).float().to(device)

    # Initial conditions at t=1
    bnd2 = torch.cartesian_prod(x, torch.from_numpy(np.array([1], dtype=np.float32))).float().to(device)

    # u(1,x)=sin(pi*x)
    bndval2 = torch.sin(np.pi * bnd2[:, 0]).float().to(device)

    # Boundary conditions at x=0
    bnd3 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float32)), t).float().to(device)

    # u(0,t)=0
    bndval3 = torch.from_numpy(np.zeros(len(bnd3), dtype=np.float32)).to(device)

    # Boundary conditions at x=1
    bnd4 = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float32)), t).float().to(device)

    # u(1,t)=0
    bndval4 = torch.from_numpy(np.zeros(len(bnd4), dtype=np.float32)).to(device)

    return bnd1, bndval1, bnd2, bndval2, bnd3, bndval3, bnd4, bndval4


def bconds_float16(x, t, grid):
    # Initial conditions at t=0
    bnd1 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float16))).half().to(device)

    # u(0,x)=sin(pi*x)
    bndval1 = torch.sin(torch.pi * bnd1[:, 0].float()).half().to(device)

    # Initial conditions at t=1
    bnd2 = torch.cartesian_prod(x, torch.from_numpy(np.array([1], dtype=np.float16))).half().to(device)

    # u(1,x)=sin(pi*x)
    bndval2 = torch.sin(np.pi * bnd2[:, 0].float()).half().to(device)

    # Boundary conditions at x=0
    bnd3 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float16)), t).half().to(device)

    # u(0,t)=0
    bndval3 = torch.from_numpy(np.zeros(len(bnd3), dtype=np.float16)).to(device)

    # Boundary conditions at x=1
    bnd4 = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float16)), t).half().to(device)

    # u(1,t)=0
    bndval4 = torch.from_numpy(np.zeros(len(bnd4), dtype=np.float16)).to(device)

    return bnd1, bndval1, bnd2, bndval2, bnd3, bndval3, bnd4, bndval4


model = torch.nn.Sequential(
    torch.quantization.QuantStub(),
    torch.nn.Linear(2, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 1),
    torch.quantization.DeQuantStub()

)
model_default = torch.nn.Sequential(

    torch.nn.Linear(2, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 1),
)

class SampleLinearModel(torch.nn.Module):

    def __init__(self):
        super(SampleLinearModel, self).__init__()
        # QuantStub converts the incoming floating point tensors into a quantized tensor
        self.quant = torch.quantization.QuantStub()
        self.linear1 = torch.nn.Linear(2, 100)
        self.tanh1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(100, 100)
        self.tanh2 = torch.nn.Tanh()
        self.linear3 = torch.nn.Linear(100, 100)
        self.tanh3 = torch.nn.Tanh()
        self.linear4 = torch.nn.Linear(100, 100)
        self.tanh4 = torch.nn.Tanh()
        self.linear5 = torch.nn.Linear(100, 1)
        # DeQuantStub converts the given quantized tensor into a tensor in floating point
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # using QuantStub and DeQuantStub operations, we can indicate the region for quantization
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.linear1(x)
        x = self.tanh1(x)
        x = self.linear2(x)
        x = self.tanh2(x)
        x = self.linear3(x)
        x = self.tanh3(x)
        x = self.linear4(x)
        x = self.tanh4(x)
        x = self.linear5(x)
        x = self.dequant(x)
        return x

model_fp32 = model

model_fp32.eval()
model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')
# model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32, [['conv', 'relu']])
quantized_model = torch.ao.quantization.prepare_qat(model_fp32.train())

quantized_model.to(device)

x, t, grid = grid32()
bnd1, bndval1, bnd2, bndval2, bnd3, bndval3, bnd4, bndval4 = bconds_float32(x, t, grid)


def wave_op(model, points):
    return torch.mean(
        (4 * nn_autograd_simple(model, points, 2, axis=0) - nn_autograd_simple(model, points, 2, axis=1)) ** 2)


def bnd_op(model):
    bnd = torch.cat((model(bnd1), model(bnd2), model(bnd3), model(bnd4)))
    bndval = torch.cat((bndval1, bndval2, bndval3, bndval4)).reshape(-1, 1)
    return torch.mean(torch.abs(bnd - bndval))


optimizer = torch.optim.Adam(model_default.parameters(), lr=1e-4)

initial_weights = model_default.state_dict()

model_default.to(device)

def closure():
    optimizer.zero_grad()
    loss = wave_op(model_default, grid) + 100 * bnd_op(model_default)
    loss.backward()
    return loss



t = 0


# allocated_memory = torch.cuda.memory_allocated()  # Общее количество памяти, выделенное для тензоров
# cached_memory = torch.cuda.memory_cached()  # Общее количество памяти, зарезервированное CUDA

# print(f'{torch.cuda.get_device_properties(0).total_memory / (1024):.2f}')
# print(f"Allocated memory: {allocated_memory / (1024):.2f} GB")
# print(f"Cached memory: {cached_memory / (1024):.2f} GB")


# start = time.time()
#
# while t < 1e4:
#     loss = optimizer.step(closure)
#     curr_loss = loss.item()
#     t += 1
#     print('t={},loss={}'.format(t, curr_loss))
#     print(GPUtil.showUtilization())
#     # print(torch.cuda.memory_summary(device=None, abbreviated=False))
#     min_loss = curr_loss
#
# end = time.time()
#
# print('time w/o amp {}'.format(end-start))

model_default.load_state_dict(initial_weights)
optimizer_amp = torch.optim.Adam(model_default.parameters(), lr=1e-4)

use_amp = True
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

start = time.time()



while t < 1e4:
    optimizer_amp.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
        loss = wave_op(model_default, grid) + 100 * bnd_op(model_default)
    scaler.scale(loss).backward()
    scaler.step(optimizer_amp)
    scaler.update()
    curr_loss = loss.item()
    t += 1
    print('t={},loss={}'.format(t, curr_loss))
    print(GPUtil.showUtilization())
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    min_loss = curr_loss

end = time.time()

print('time w amp {}'.format(end-start))

# if mixed_precision:
#     print("Mixed-precision mode enabled. Works only with GPU.")
#     optimizer.zero_grad()
#     t = 0
#     while t < 1e4:
#         with torch.autocast(device_type=device, dtype=torch.float16, enabled=mixed_precision):
#             loss, loss_normalized = Solution_class.evaluate(second_order_interactions=second_order_interactions,
#                                                             sampling_N=sampling_N,
#                                                             lambda_update=lambda_update,
#                                                             tol=tol)
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         curr_loss = loss_normalized.item()
#         t += 1
#         if t % 500 == 0:
#             print('t={},loss={}'.format(t, curr_loss))
#     return self.model