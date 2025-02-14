import argparse
from pathlib import Path

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='uci-msg', help='Dataset')

    parser.add_argument('--cuda_device', type=int,
                        default=0, help='Cuda device no -1')

    parser.add_argument('--seed', type=int, default=2023, help='split seed')

    parser.add_argument('--repeat', type=int, default=5, help='number of repeat model')

    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train.')

    parser.add_argument('--out_dim', type=int, default=64,
                        help='model output dimension.')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type')

    parser.add_argument('--lr', type=float, default=0.01,
                        help='initial learning rate.')

    parser.add_argument('--maml_lr', type=float, default=0.008,
                        help='meta learning rate')

    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (L2 loss on parameters).')

    parser.add_argument('--num_layers', type=int,
                        default=2, help='GNN layer num')

    parser.add_argument('--num_hidden', type=int, default=256,
                        help='number of hidden units of MLP')

    parser.add_argument('--window_num', type=float, default=8,
                        help='windows size')

    parser.add_argument('--dropout', type=float, default=0.1,
                        help='GNN dropout')

    parser.add_argument('--residual', type=bool, default=False,
                        help='skip connection')

    parser.add_argument("--dropout_1", type=float, default=0.2,
                        help="droupout for GRU and cross attention")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="hidden and embed dimention for GRU and cross attention")
    parser.add_argument("--seq_num_layer", type=int, default=2,
                        help="GRU sequence block")

    parser.add_argument("--num_heads", type=int, default=1,
                        help="cross attention head")
    parser.add_argument("--light_gru", type=bool, default=False,
                        help="To switch between normal and light GRU")
    

    roland_data = parser._option_string_actions["--dataset"].default
    path_1 = str(Path(__file__).parent.parent)
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file path',
        default=f"{path_1}/roland-master/run/roland_example_{roland_data}.yaml",
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
        default=f"{path_1}/roland-master/roland_public_data",
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