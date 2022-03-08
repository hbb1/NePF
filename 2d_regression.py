import jax
from jax import random, grad, jit, vmap
from jax.config import config
import jax.numpy as np
from jax.experimental import optimizers, stax
# from livelossplot import PlotLosses
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm as tqdm
import os
import imageio
# from tqdm.notebook import tqdm as tqdm
from tqdm import tqdm
import numpy as onp
import pdb
from PIL import Image

def make_network(num_layers, num_channels):
    layers = []
    for i in range(num_layers-1):
        layers.append(stax.Dense(num_channels))
        layers.append(stax.Relu)
    layers.append(stax.Dense(1))
    layers.append(stax.Sigmoid)
    return stax.serial(*layers)

model_loss = jit(lambda params, x, y: .5 * np.mean((apply_fn(params, x) - y) ** 2))
model_psnr = jit(lambda params, x, y: -10 * np.log10(2.*model_loss(params, x, y)))
model_grad_loss = jit(lambda params, x, y: jax.grad(model_loss)(params, x, y))
input_encoder = jit(lambda x, a, b: np.concatenate([a * np.sin((2.*np.pi*x) @ b.T), 
                                                    a * np.cos((2.*np.pi*x) @ b.T)], axis=-1))

def train_model(lr, iters, train_data, test_data):
    opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_update = jit(opt_update)

    _, params = init_fn(rand_key, (-1, train_data[0].shape[-1]))
    opt_state = opt_init(params)

    train_psnrs = []
    test_psnrs = []
    xs = []
    for i in tqdm(range(iters), desc='train iter', leave=False):
        opt_state = opt_update(i, model_grad_loss(get_params(opt_state), *train_data), opt_state)
        if i % 25 == 0:
            train_psnr = model_psnr(get_params(opt_state), *train_data)
            test_psnr = model_psnr(get_params(opt_state), *test_data)
            train_psnrs.append(train_psnr)
            test_psnrs.append(test_psnr)
            xs.append(i)

    results = {
        'state': get_params(opt_state),
        'train_psnrs': train_psnrs,
        'test_psnrs': test_psnrs,
        'xs': xs
    }
    return results


## Random seed
rand_key = random.PRNGKey(10)
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
data = np.array(Image.open('MRI.bmp'))[..., None] / 255


network_depth =  4#@param
network_width = 256 #@param
lr =  1e-3#@param
training_steps =  2000#@param
test_scales =  [13,14,15]#@param
embedding_size =  256#@param
init_fn, apply_fn = make_network(network_depth, network_width)

y_train = data
x1 = np.linspace(0, 1, data.shape[0]//2+1)[:-1]
x_train = np.stack(np.meshgrid(x1,x1), axis=-1)
x1_t = np.linspace(0, 1, data.shape[0]+1)[:-1]
x_test = np.stack(np.meshgrid(x1_t,x1_t), axis=-1)
bvals = random.normal(rand_key, (embedding_size, 2))
avals = np.ones((bvals.shape[0]))
scale = 10
train_data = (input_encoder(x_train, avals, bvals*scale), y_train[::2,::2,:])
test_data = (input_encoder(x_test[1::2,1::2], avals, bvals*scale), y_train[1::2,1::2,:])
results = train_model(lr, training_steps, train_data, test_data)
image = apply_fn(results['state'], test_data[0])
pdb.set_trace()