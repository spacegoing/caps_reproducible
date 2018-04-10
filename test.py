import numpy as np

# Test Batch Inputs

b1_caps1 = np.load("./tmp/dt/img.npy")
b2_caps1 = np.load("./tmp/dt/img_1.npy")

print(np.max(b1_caps1))
print(np.max(b2_caps1))
print(np.min(b1_caps1))
print(np.min(b2_caps1))

diff = np.abs(b1_caps1 - b2_caps1)

max_ind = np.unravel_index(np.argmax(diff, axis=None), diff.shape)
print(max_ind)
min_ind = np.unravel_index(np.argmin(diff, axis=None), diff.shape)
print(min_ind)

print(np.max(diff))
print(np.min(diff))
print(np.mean(diff))


# Test Capsule Activations
act1_caps1 = np.load("./tmp/dt/conv_capsule1_activations.npy")
act2_caps1 = np.load("./tmp/dt/conv_capsule1_activations_1.npy")

print(np.max(act2_caps1))
print(np.max(act2_caps1))
print(np.min(act1_caps1))
print(np.min(act1_caps1))

diff = np.abs(act2_caps1 - act1_caps1)

max_ind = np.unravel_index(np.argmax(diff, axis=None), diff.shape)
print(max_ind)
min_ind = np.unravel_index(np.argmin(diff, axis=None), diff.shape)
print(min_ind)

print(np.max(diff))
print(np.min(diff))
print(np.mean(diff))
