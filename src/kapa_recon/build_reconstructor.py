from typing import Optional, Sequence, Tuple
from numpy.typing import NDArray
import pyrao as rao
import numpy as np
import matplotlib.pyplot as plt
from kapa_recon.expose_reconstructor import load_measured_recon
import os
import argparse
import datetime

# Optimisable parameters
D_REG = float(os.environ.get("D_REG", "1e-4"))
C_REG = float(os.environ.get("C_REG", "1e2"))
TTF = bool(os.environ.get("TTF", "1") == "1")

# Constants
NSUBAP: int = 304


def save_recon(recon: np.ndarray, filename: str):
    bytes = recon.astype(dtype=np.dtype("float32").newbyteorder(">")).tobytes()
    with open(filename, "wb") as f:
        f.write(bytes)
    print(f"saved reconstructor to {filename}")


def compose_matrix(args: Tuple[Sequence[float], Tuple[int, int]]) -> NDArray:
    return np.array(args[0]).reshape(args[1])


def main(cmm_reg_diag: Optional[NDArray] = None) -> NDArray:
    parser = argparse.ArgumentParser(
        "build-cmat",
        description="""
Build an AO reconstructor from a specified AO system definition.
""",
    )
    parser.add_argument(
        "system",
        help='system parameters .yaml file, usually output by "python -m kapa_recon.fit_imat"',
    )
    parser.add_argument(
        "--prefix",
        "-p",
        help='filename prefix to save control matrix to, default: "./<UTC-DATETIME>"',
        default=f"./{datetime.datetime.now(datetime.timezone.utc).strftime(r"%Y%m%dT%H%M%SZ")}",
    )
    parser.add_argument(
        "--plot",
        help="filename to save comparison figure if desired",
    )

    args = parser.parse_args()

    expanded_system = rao.ExpandedSystem.load_yaml(args.system)
    cmm = compose_matrix(expanded_system.c_meas_meas())
    dtc = compose_matrix(expanded_system.d_ts_com())
    ctm = compose_matrix(expanded_system.c_ts_meas())
    dmc = compose_matrix(expanded_system.d_meas_com())
    pm = np.array(expanded_system.p_meas())

    tt_filter_block = np.eye(NSUBAP) - np.ones([NSUBAP, NSUBAP]) / NSUBAP
    tt_filter = np.concat(
        [
            np.concat(
                [
                    tt_filter_block if i == j else tt_filter_block * 0.0
                    for i in range(2 * 4)
                ],
                axis=0,
            )
            for j in range(2 * 4)
        ],
        axis=1,
    )

    print(
        f"unbiased cmm diag mean/std: {cmm.diagonal().mean():0.3e} "
        f"+/- {cmm.diagonal().std():0.3e} rms"
    )

    dcc = dtc.T @ dtc
    print(
        f"unbiased dcc diag mean/std: {dcc.diagonal().mean():0.3e} "
        f"+/- {dcc.diagonal().std():0.3e} rms"
    )

    if cmm_reg_diag is None:
        cmm += C_REG * np.eye(cmm.shape[0])
    else:
        cmm += np.diag(cmm_reg_diag)
    if TTF:
        cmm = tt_filter @ cmm @ tt_filter.T
    if TTF:
        ctm = ctm @ tt_filter.T
    dcc += D_REG * np.eye(dtc.shape[1])
    if TTF:
        dmc = tt_filter @ dmc
    piston_filter = np.eye(dmc.shape[1]) - np.ones([dmc.shape[1]] * 2) / dmc.shape[1]

    rcm = -(piston_filter @ np.linalg.solve(cmm, np.linalg.solve(dcc, dtc.T @ ctm).T).T)
    if TTF:
        rcm = rcm @ tt_filter
    # borrow the bottom few rows from the measured reconstructor
    measured_recon = load_measured_recon()
    full_recon = np.concat([rcm, measured_recon[-11:, :]], axis=0)
    descriptor = f"{'no' if TTF else ''}-tt_{D_REG:0.1e}-dreg_{C_REG:0.1e}-creg"

    if args.plot:
        plt.figure(figsize=[12, 4])
        plt.imshow(full_recon)
        plt.colorbar()
        plt.xlabel("measurements")
        plt.ylabel("actuators")
        plt.tight_layout()
        plt.savefig(f"{args.plot}.png", dpi=300)

    save_recon(full_recon, f"{args.prefix}_{descriptor}.mr")
    return full_recon


if __name__ == "__main__":
    main()
