from pathlib import Path
from get_args import load_args_wingnn, load_args_haks
import os

if __name__ == '__main__':
    model_type = "wingnn"
    path = f"{Path(__file__).parent}"
    path_1 = f"{path}/final_results"
    if not os.path.exists(path_1):
          os.mkdir(path_1)
    if model_type == "wingnn":
        from main_models.main_wingnn_settings import main_wingnn
        args = load_args_wingnn(path_1)
        main_wingnn(args, path, path_1,  model_type)
    elif model_type == "hawkes":
        from main_models.main_hawkes_settings import main_hawkes
        args = load_args_haks()
        main_hawkes(args, path_1, model_type)