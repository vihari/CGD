# CGD: A simple algotihm for robustness to sub-population shifts   
**Under review at ICLR 2022**   

## Instructions for running    
The code is built on [WILDS codebase v1.2.2](https://github.com/p-lambda/wilds/releases/tag/v1.2.2) and run on TPU v3-8. For efficiency, we only release the algorithm files and detail the minimal edits to be made on the WILDS codebase below. 

1. Move the python files under algorithms to `examples/algorithms` of the WILDS codebase.   
2. Edit `examples/algorithms/initializer.py` to add an import and initialization line as follows:   
```python
  from algorithms.CG import CG
  .....
  .....
  elif config.algorithm.startswith('CG'):
      train_g = train_grouper.metadata_to_group(train_dataset.metadata_array)
      is_group_in_train = get_counts(train_g, train_grouper.n_groups) > 0
      groups, g_counts = np.unique(train_g, return_counts=True)
      alg = CG
      algorithm = alg(
          config=config,
          d_out=d_out,
          grouper=train_grouper,
          loss=loss,
          metric=metric,
          n_train_steps=n_train_steps,
          is_group_in_train=is_group_in_train,
          group_info=[groups, g_counts]
      )
```
3. Add default configuration of the algorithms to `configs/algorithm.py` such as the following:
```python
   'CG': {
        'train_loader': 'standard',
        'uniform_over_groups': True,
        'distinct_groups': True,
        'eval_loader': 'standard',
        'cg_step_size': 0.1
    },
```
4. Add the algorithm name to algorithms variable in `examples/configs/supported.py`.
5. Finally add to `examples/run_expt.py` the lines below.
```python
parser.add_argument('--cg_C', type=float, default=0)
parser.add_argument('--cg_step_size', type=float, default=0.05)
parser.add_argument('--pgi_penalty_weight', type=float)
```

After the edits above, we can run using:   
```bash
python run_expt.py --dataset $DATASET --algorithm $ALG --root_dir data --progress_bar --log_dir logs/"$DATASET"/$ALG/run:1:seed:"$seed" --seed $seed --weight_decay 1e-4 --n_epochs 100;
```

## File structure
In the algorithms folder, we include implementation of our method (CGD) and Ahmed et.al. ICLR 2021.   
1. CGD: `algorithms/cg.py`. Hyperparameters: `--cg_step_size` set the step size parameter and `--cg_C` sets the group adjustment parameter C discussed in our paper. 
2. PGI ([ICLR 2021](https://openreview.net/forum?id=b9PoimzZFJ)): `algorithms/pgi.py`. Hyperparameter: `--pgi_penalty_weight` controls ths weight of distributional divergence discussed in their paper (lambda).

## Datasets
Most of the datasets can be readily downloaded by passing `--download` when running `run_expt.py`. Additional datasets that are not part of WILDS are included in the datasets folder, these include Colored-MNIST (`cmnist_debug_dataset.py`) and datasets used for qualitative evaluation of Section 4 from our paper. 
