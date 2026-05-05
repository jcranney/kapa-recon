import glob
import pyrao
import numpy as np
import matplotlib.pyplot as plt
from expose_reconstructor import load_measured_recon


def save_recon(recon: np.ndarray, filename: str):
    bytes = recon.astype(dtype=np.dtype("float32").newbyteorder(">")).tobytes()
    with open(filename, "wb") as f:
        f.write(bytes)


if __name__ == "__main__":
    system_geoms = []
    for filename in glob.glob("output_data/LGS*.yaml"):
        system_geoms.append(pyrao.SystemGeom.load_yaml(filename))
    system_geom = pyrao.SystemGeom.merge_meas(system_geoms)
    recon_matrices = pyrao.ReconMatrices.new(system_geom)

    NSUBAP: int = 304
    TTF: bool = True
    tt_filter_block = np.eye(NSUBAP) - np.ones([NSUBAP, NSUBAP])/NSUBAP
    tt_filter = np.concat(
        [
            np.concat(
                [
                    tt_filter_block if i==j else tt_filter_block*0.0
                    for i in range(2*4)
                ],
                axis=0,
            )
            for j in range(2*4)
        ],
        axis=1,
    )
    plt.matshow(tt_filter)
    plt.colorbar()
    plt.savefig("output_data/ttfilter.png",dpi=300)
    plt.close()
    
    cmm = np.array(recon_matrices.c_meas_meas)
    print(f"unbiased cmm diag mean/std: {cmm.diagonal().mean():0.3e} "
          f"+/- {cmm.diagonal().std():0.3e} rms")
    
    dtc = np.array(recon_matrices.d_ts_com)
    dcc = dtc.T @ dtc
    print(f"unbiased dcc diag mean/std: {dcc.diagonal().mean():0.3e} "
          f"+/- {dcc.diagonal().std():0.3e} rms")

    D_REG: float = 1e-4
    C_REG: float = 1e2

    cmm += C_REG * np.eye(cmm.shape[0])
    if TTF:
        cmm = tt_filter @ cmm @ tt_filter.T
    ctm = np.array(recon_matrices.c_ts_meas)
    if TTF:
        ctm = ctm @ tt_filter.T
    dcc += D_REG*np.eye(dtc.shape[1])
    dmc = np.array(recon_matrices.d_meas_com)
    if TTF:
        dmc = tt_filter @ dmc
    pm = np.array(recon_matrices.p_meas)
    piston_filter = np.eye(dmc.shape[1]) - \
        np.ones([dmc.shape[1]]*2)/dmc.shape[1]

    rcm = -(piston_filter @ np.linalg.solve(cmm,np.linalg.solve(dcc, dtc.T @ ctm).T).T)
    if TTF:
        rcm = rcm @ tt_filter 
    # borrow the bottom few rows from the measured reconstructor
    measured_recon = load_measured_recon()
    full_recon = np.concat(
        [rcm, measured_recon[-11:, :]],
        axis=0
    )
    
    plt.figure(figsize=[12, 4])
    plt.imshow(full_recon)
    plt.colorbar()
    plt.xlabel("measurements")
    plt.ylabel("actuators")
    plt.tight_layout()
    plt.savefig(f"./output_data/recon_{'filt' if TTF else 'keep'}-tt.png", dpi=300)
    save_recon(full_recon, f"./output_data/16Apr0005_jcr_{'filt' if TTF else 'keep'}-tt.mr")