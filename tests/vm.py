import torch
import torch.distributions as D
import numpy as np
import matplotlib.pyplot as plt
from utils.plotting_utils import add_distribution_to_ax, add_scatter_to_ax

batch_size = 10
N = 5
extra_dim = 2
original_angles = [np.linspace(0, 1, N, endpoint=False) * 2 * np.pi] * batch_size
location = torch.tensor(np.stack([np.cos(original_angles), np.sin(original_angles)], axis=-1))
angle = torch.atan2(location[:, :, 1], location[:, :, 0])
var = torch.exp(torch.tensor(-4.5) * torch.ones((batch_size, N, extra_dim)))
print(location.shape, var.shape)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].set_aspect('equal', adjustable='box')
axes[1].set_aspect('equal', adjustable='box')
# Plot the distribution for the first batch
add_distribution_to_ax(location[0], var[0], axes[0], N)
axes[0].set_xlim(-1.2, 1.2)
axes[0].set_ylim(-1.2, 1.2)
# plt.show()


# print(angle, angle.shape)
concentration = 1 / var[..., -1]
# print(concentration)
d = D.von_mises.VonMises(angle, concentration)
d2 = D.Independent(d, 1)
mix = D.Categorical(torch.ones((batch_size, N)))

print("Event shape", d.event_shape, "Batch shape", d.batch_shape)
print("Event shape", d2.event_shape, "Batch shape", d2.batch_shape)
print("Event shape", mix.event_shape, "Batch shape", mix.batch_shape)
m = D.MixtureSameFamily(mix, d)
print("Event shape", m.event_shape, "Batch shape", m.batch_shape)
samples = m.sample((10,)).reshape((-1))
samples = torch.tensor(np.stack([np.cos(samples), np.sin(samples)], axis=-1))

print(samples.shape)
add_scatter_to_ax(samples, axes[1], color="red")
axes[1].set_xlim(-1.2, 1.2)
axes[1].set_ylim(-1.2, 1.2)
plt.show()
