import scipy.optimize as opt
import time
from devtools import pprint  # type: ignore
import matplotlib.pyplot as plt
from disp_imat import read_array, write_array
import glob
from typing import Tuple, List
from kapa import Perturbations, build_imat, build_system
import numpy as np

# define keck WFSs
lgs_wfss: List[Tuple[float,float]] = [
    (17.5 * np.cos(theta), 17.5 * np.sin(theta))
    for theta in np.arange(4) * 2 * np.pi / 4
]

def cost_vector(pert, imat_true, lgs_dir: Tuple[float,float]):
    """
    We will use a least squares solver, so can build a cost vector of the
    signed residuals.
    """
    err = (
        imat_true - build_imat(Perturbations.from_list(pert), lgs_dir, indices)
    ).flatten()
    return err

if __name__ == "__main__":
    for filename in glob.glob("./input_data/24.imx-LGS*"):
        name = filename.split("-")[-1]
        print(f"Fitting {name} measured imat")
        idx = int(name[-1])-1
        imat_true = read_array(filename)
        indices = None  # can filter by imat_true values later if too slow

        # run optimisation
        t1 = time.time()
        # pert_initial = np.zeros(NPARAM_DM * NDM + NPARAM_WFS * NWFS)
        pert_initial = Perturbations()
        result = opt.least_squares(
            lambda pert: cost_vector(pert, imat_true, lgs_wfss[idx]),
            pert_initial.to_list(),
            method="trf",  # both "lm" and "trf" seem to
            # work equally fast and accurately
            verbose=2
        )
        t2 = time.time()

        # print results
        print(f"took {t2 - t1:0.2f} seconds")
        print(result)
        print("estimate:")
        perturbations = Perturbations.from_list(result["x"])
        pprint(perturbations)
        system_geom = build_system(perturbations, lgs_dir=lgs_wfss[idx])
        system_geom.save_yaml(f"./output_data/{name}.yaml")
        # perturbations.save(filename.split("-")[-1]+".json")
        imat_est = build_imat(perturbations, lgs_wfss[idx], indices=None)
        fig, ax = plt.subplots(1, 3, figsize=(10, 4))
        clim = (imat_true.min(), imat_true.max())
        for a, im in zip(ax, [imat_true, imat_est, imat_true-imat_est]):
            a.imshow(im, vmin=clim[0], vmax=clim[1])
        plt.tight_layout()
        plt.savefig(f"./output_data/{name}.png", dpi=300)
        write_array(f"./output_data/24.simx-{name}", imat_est)