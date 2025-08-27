import argparse
from pathlib import Path

def load_args_wingnn(path_1):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='uci-msg', help='Dataset')

    parser.add_argument('--cuda_device', type=int,
                        default=1, help='Cuda device no -1')

    parser.add_argument('--seed', type=int, default=2023, help='split seed')

    parser.add_argument('--repeat', type=int, default=2, help='number of repeat model')

    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train.')
    parser.add_argument('--out_dim', type=int, default=64,
                        help='model output dimension.')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type')


    parser.add_argument('--num_layers', type=int,
                        default=2, help='GNN layer num')

    parser.add_argument('--num_hidden', type=int, default=128,
                        help='number of hidden units of MLP')

    parser.add_argument('--dropout', type=float, default=0.2,
                        help='GNN dropout')

    parser.add_argument('--residual', type=bool, default=True,
                        help='skip connection')

    parser.add_argument("--dropout_1", type=float, default=0.2,
                        help="droupout for GRU and cross attention")
    
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="hidden and embed dimention for GRU and cross attention")
    

    parser.add_argument("--num_heads", type=int, default=1,
                        help="cross attention head")

    

    roland_data = parser._option_string_actions["--dataset"].default
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file path',
        default=f"{path_1}/roland/run/roland_example_{roland_data}.yaml",
        type=str
    )
    parser.add_argument(
        '--mark_done',
        dest='mark_done',
        action='store_true',
        help='mark yaml as yaml_done after a job has finished',
    )

    parser.add_argument(
        '--override_remark',
        dest='override_remark',
        type=str,
        default="roland_example",
        help='easily override the remark in the yaml file'
    )

    parser.add_argument(
        '--override_data_dir',
        dest='override_data_dir',
        type=str,
        required=False,
        default=f"{path_1}/roland/roland_public_data",
        help='easily override the dataset.dir in the yaml file'
    )

    parser.add_argument(
        'opts',
        help='See graphgym/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )

    args = parser.parse_args()
    return args


def load_args_haks():
    parser = argparse.ArgumentParser(description='')
    # run configuration
    parser.add_argument('--dataset', type=str, default='uci')
    parser.add_argument('--model', type=str, default='DyGSSM', choices=['gcn', 'gat', 
                        'hgcn', 'hgat', 'dysat', 'evolve-o', 'evolve-h', 'lstmgcn', 'wdgcn',
                        'vgrnn', 'roland', 'wingnn', 'DyGSSM', 'htgn', 'graphmixer', 'm2dne', 'ghp']) # 
    parser.add_argument('--node_feat', type=str, default='dummy', choices=['onehot-id', 'dummy'])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--patiance', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=1) # 200
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runs', type=int, default=1) # 1
    parser.add_argument('--no_log', action="store_true")
    parser.add_argument('--row_mrr', action="store_true")
    
    # general model configuration
    parser.add_argument('--n_neg_train', type=int, default=1)
    parser.add_argument('--n_neg_test', type=int, default=100)
    parser.add_argument('--window', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_hidden', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    
    # hawkes gnn
    parser.add_argument('--bias', action="store_true") # bias in layer
    parser.add_argument('--bn', action="store_true")
    parser.add_argument('--time_encoder', action="store_true") # bias in layer
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--norm_type', type=str, default='snorm', choices=['snorm', 'dnorm', 'hnorm'])
    
    # roland
    parser.add_argument('--roland_updater', type=str, default='ma', choices=['gru', 'mlp', 'ma', 'gru-ma']) 
    parser.add_argument('--roland_is_meta', action="store_true")
    parser.add_argument('--roland_alpha', type=float, default=0.9)

    # wingnn
    parser.add_argument('--wingnn_maml_lr', type=float, default=0.008)
    parser.add_argument('--wingnn_drop_snap', type=float, default=0.1)

    # test
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--minibatch', action="store_true")
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--exp_name', type=str, default='')

    return parser.parse_args()