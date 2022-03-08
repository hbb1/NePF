import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence
import optax
import matplotlib.pyplot as plt
import numpy as np
import pdb
import skimage.transform 
from cv2 import cv2
from PIL import Image
import scipy.interpolate as intp

def posenc(x, max_deg):
    scales = jnp.array([2**i for i in range(0, max_deg)])
    xb = jnp.reshape((x[Ellipsis, None, :] * scales[:, None]),
                     list(x.shape[:-1]) + [-1])
    four_feat = jnp.sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
    return jnp.concatenate([four_feat], axis=-1)

class MLP(nn.Module):
    features: Sequence[int]
    @nn.compact
    def __call__(self, x):
        x = posenc(x, 3)
        for feat in self.features[:-1]:
            x = nn.relu((nn.Dense(feat))(x))
        x = nn.Dense(self.features[-1])(x)
        return nn.sigmoid(x)

# @jax.jit
def train_step(model, params, opt_state, batch):
    def loss_fn(params):        
        # batch = np.stack((fxl, fyl), axis=-1)
        Fx = model.apply(params, batch['coord'])
        loss = np.mean((batch['color'] - Fx)**2)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, loss, opt_state


model = MLP([2, 128, 128, 128, 1])
im = Image.open('MRI.bmp')
im = np.array(im)
x, y = np.meshgrid(np.linspace(0,1,im.shape[0]), np.linspace(0,1,im.shape[0]))
coords = np.stack((x,y), axis=-1)
variables = model.init(jax.random.PRNGKey(0), coords)

# init optimizer
tx = optax.adam(1e-3)
params = variables
opt_state = tx.init(params)
im = im / 255
acc_loss = []
print_iter = 100
test_iter = 10000
for i in range(100000):
    batch = dict(coord=coords, color=im)
    params, loss, opt_state = train_step(model, params, opt_state, batch)
    # acc_loss.append(loss)
    # if i+1 % print_iter == 0:
        # pdb.set_trace()
    print("iter {} \t loss {}".format(i, loss))
    if i+1 % test_iter == 0:
        # testing
        Fx = model.apply(params, batch['coord'])
        plt.imshow(Fx[..., 0])
        plt.savefig('figure_{}'.format(i))

pdb.set_trace()