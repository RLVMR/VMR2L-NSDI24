## 	Virtual Machine ReScheduling Based on Reinforcement Learning

### installation steps

1.  Install conda:

```
$ conda create -n vm python=3.9
$ conda activate vm
```

2. Install rllib:

```
$ pip install gym==0.23.1
$ pip install "ray[rllib]" tensorflow torch
$ pip install -e gym-reschdule_combination
$ pip install tqdm matplotlib pandas wandb
```

3. Optional:
```
$ conda install -c dglteam dgl-cuda11.3
```

### Run RL

- Training
```
$ python3 main.py
```
- To use wandb, first get your [API key](https://wandb.ai/authorize) from wandb 
```
$ wandb login
$ python3 main.py --track --model [mlp/attn]
```
- To use pretrained model for VM selection
```
$ python3 main.py --track --model [mlp/attn] --pretrain
```
- Evaluation
```
$ python3 eval.py --restore-name [] --restore-file-name [] --model [mlp/attn]
```

### Environments
* generalizer-v0: Base environment. Fixed number of VMs.
* generalizer-v1: Dynamic number of VMs.
* generalizer-v2: Allow agent to choose to deploy on NUMA 0 or 1.
* generalizer-v3: Output detailed statistics during step.
* graph-v1: Dynamic number of VMs with vm-pm affiliations to support graph models.
* graph-v3: add flip support.
* graph-v4: include PM info in VM covariates.
