from numpy.typing import NDArray
import scipy.optimize as opt
import time
from devtools import pprint
import matplotlib.pyplot as plt
from kapa_recon.disp_imat import read_array, write_array
import glob
from typing import Sequence
from kapa_recon.kapa import Perturbations, build_imat, kapa
import numpy as np
import argparse


def cost_vector(pert: Sequence[float], imat_true: NDArray) -> NDArray:
    """
    We will use a least squares solver, so can build a cost vector of the
    signed residuals.
    """
    pert_ob = Perturbations.from_list(pert)
    system = kapa(pert_ob)
    err = (imat_true - build_imat(system)).flatten()
    return err

def main():
    parser = argparse.ArgumentParser(
        "fit-imat",
        description="""
Optimise parameters of a synthetic LGS interaction matrix model to fit a measured
LGS interaction matrix, for the Keck KAPA system.
""")
    parser.add_argument(
        "imats",
        help='glob pattern for LGS interaction matrices, e.g.: "./24.imx-LGS*"',
    )
    parser.add_argument(
        "--output",
        "-o",
        help='yaml file for saving fitted system parameters, default: "./ao-system.yaml"',
        default="./ao-system.yaml",
    )
    parser.add_argument(
        "--plot",
        "-p",
        help="filename to save comparison figure to, if desired",
    )
    parser.add_argument(
        "--save",
        "-s",
        help="filename to save the fitted iMat to, if desired.",
    )

    args = parser.parse_args()

    imat_true = np.zeros([0, 349])
    for filename in glob.glob(args.imats):
        x = read_array(filename)
        imat_true = np.concat([imat_true, x])

    # run optimisation
    t1 = time.time()
    pert_initial = Perturbations()

    def cost_fun(pert: NDArray) -> NDArray:
        return cost_vector(list(pert), imat_true)

    result = opt.least_squares(
        cost_fun,
        pert_initial.to_list(),
        method="trf",  # both "lm" and "trf" seem to
        # work equally fast and accurately
        verbose=2,
        max_nfev=20,
    )
    t2 = time.time()

    # print results
    print(f"took {t2 - t1:0.2f} seconds")
    print(result)
    print("estimate:")
    perturbations = Perturbations.from_list(result["x"])
    # perturbations.save("./output_data/perturbations.json")
    pprint(perturbations)
    expanded_system = kapa(perturbations)
    expanded_system.save_yaml(args.output)
    imat_est = build_imat(expanded_system)

    if args.save:
        write_array(args.save, imat_est)

    if args.plot:
        fig, ax = plt.subplots(1, 3, figsize=(10, 10))
        clim = (imat_true.min() / 10.0, imat_true.max() / 10.0)
        for a, im in zip(ax, [imat_true, imat_est, imat_true - imat_est]):
            a.imshow(im, vmin=clim[0], vmax=clim[1])
        plt.tight_layout()
        plt.savefig(args.plot, dpi=300)
        plt.close()

if __name__ == "__main__":
    main()