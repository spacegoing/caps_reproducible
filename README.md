# Run & Debug #

- `tmpy t` : train caps net
- `tmpy e` : evaluate caps net (loading last step from training
  checkpoint dir `tmp/train/` to evaluate)
- `tmpy d` : debug caps net with tfdbg

### tfdbg workflow ###

workflow of dumping tensors using `tfdbg`:

1. execute `tmpy d` in command line to invoke `tfdbg`
2. in `tfdbg` execute `run` command to execute first session run
3. execute `pt tower_0/Reshape:0 -w tmpy/dt/img.npy` to write
   tensor into a numpy file. 
   - `tower_0/Reshape:0` is the node name of `image_4d` variable
   in `model/capsule_model.py` (line 179), which is the batch
   used in current run

repeat one more time to save `pt tower_0/Reshape:0 -w
tmpy/dt/img_1.npy` for test.

### test random seed ###

`test.py` is a simple script to load and test whether saved numpy
files are deterministic.

# Issues #

## Reproducible Result ##

I followed these documentation for fixing random seed:

- [tensorflow document](https://www.tensorflow.org/api_docs/python/tf/set_random_seed) 
- [stackoverflow issue1](https://gist.github.com/tnq177/ce34bcf6b20243b0b5b23c78833e7945)

And this for fixing batch

- [stackoverflow batch](https://stackoverflow.com/questions/48156405/tensorflow-shuffle-batch-non-deterministic)

(`shuffle_batch` has been replaced to `batch()` in
`input_data/mnist/mnist_input_record.py` `def inputs():` but
samples in batch are still random)

### Details ###

#### Entry File: `experiment.py` ####

- Both tensorflow and numpy random seed has been set
``` python
fix_seed = 1
np.random.seed(fix_seed)
```

- default graph has been reseted & graph level random seed has
  been set before session is created

```python
  tf.reset_default_graph()
  with tf.Graph().as_default():
    if FLAGS.fix_seed:
      tf.set_random_seed(fix_seed)
      print("random seed is fixed to %d" % fix_seed)
    # Build model
    features = get_features('train', 128, num_gpus, data_dir, num_targets,
                            dataset, validate)
```

