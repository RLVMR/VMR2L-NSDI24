# Federated Learning + Time Series Forecasting (2021)
Author: 

## To run:

1. Install all dependencies listed in requirements.txt. Note that the model has only been tested in the versions shown in the text file.

2. Start training. See the beginning of '__main__' for a list of arguments it takes.
  
   ```bash
   python train.py --dataset [elect_coarse] --dataloader [dec_only] --model [dec_only_quantile]
   ```
3. Evaluate a set of saved model weights:
        
   ```bash
   python evaluate.py --dataset [elect_coarse] --dataloader [dec_only] --model [dec_only_quantile] --restore-file ['best']
   ```
4. Perform hyperparameter search. Specify the range of each hyperparameter to be search on using a dictionary of list on line 110.
        
   ```bash
    python search_params.py --dataset [elect_coarse] --dataloader [dec_only] --model [dec_only_quantile] --gpu-ids [0 1 2 3]
   ```