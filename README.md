# Run & Debug #

- `tmpy t` : train caps net
- `tmpy e` : evaluate caps net (loading last step from training
  checkpoint dir `tmp/train/` to evaluate)
- `tmpy d` : debug caps net with tfdbg

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

