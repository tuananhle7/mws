import torch
import numpy as np
import sweep
import util


def main():
    test_algorithms = ["rws", "vimco", "sleep"]
    mode = "test"

    # loading files
    for sweep_args in sweep.get_sweep_argss():
        util.logging.info(util.get_path_base_from_args(sweep_args))
        log_ps = {}
        lossess = {}
        for test_algorithm in test_algorithms:
            save_path = util.get_logp_path(sweep_args, mode, test_algorithm)
            try:
                log_p, losses = torch.load(save_path)
                log_ps[test_algorithm] = log_p
                lossess[test_algorithm] = losses
            except FileNotFoundError:
                util.logging.info("skipping {}".format(save_path))
        util.logging.info(f"log_ps = {log_ps}")
        log_ps_values = list(log_ps.values())
        util.logging.info(
            f"best log_p ({test_algorithms[np.argmax(log_ps_values)]}): {np.max(log_ps_values)}"
        )
        print("-------------")


if __name__ == "__main__":
    main()
