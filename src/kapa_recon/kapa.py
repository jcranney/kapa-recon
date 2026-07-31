import pyrao as rao
import numpy as np
from typing import List, Sequence, Tuple
from numpy.typing import NDArray
from dataclasses import dataclass, fields
import json


@dataclass
class Perturbations:
    dm_coupling: float = 0.3
    dm_mpv: float = 1.0  # microns per volt
    wfs1_delta_x: float = 0.0
    wfs1_delta_y: float = 0.0
    wfs1_clocking: float = 0.0
    wfs1_zoom: float = 1.0
    wfs2_delta_x: float = 0.0
    wfs2_delta_y: float = 0.0
    wfs2_clocking: float = 0.0
    wfs2_zoom: float = 1.0
    wfs3_delta_x: float = 0.0
    wfs3_delta_y: float = 0.0
    wfs3_clocking: float = 0.0
    wfs3_zoom: float = 1.0
    wfs4_delta_x: float = 0.0
    wfs4_delta_y: float = 0.0
    wfs4_clocking: float = 0.0
    wfs4_zoom: float = 1.0

    @staticmethod
    def from_list(pert_list: Sequence[float]):
        kwargs = {}
        for i, field in enumerate(fields(Perturbations)):
            kwargs[field.name] = pert_list[i]
        return Perturbations(**kwargs)

    def to_list(self) -> Sequence[float]:
        pert: List[float] = []
        for field in fields(self):
            pert += [getattr(self, field.name)]
        return pert

    def save(self, filename: str):
        with open(filename, "w") as f:
            json.dump(self.to_list(), f)

    @staticmethod
    def load(filename: str):
        with open(filename, "r") as f:
            perturb_str = json.load(f)
        return Perturbations(perturb_str)


def kapa(pert: Perturbations) -> rao.ExpandedSystem:
    lgs_rad = 10.0 * 4.848e-6
    wfs_dirs = [
        (lgs_rad * np.cos(theta), lgs_rad * np.sin(theta))
        for theta in np.arange(4) * np.pi / 2.0
    ]
    telescope = rao.Telescope(teldiam=8.0, cobs=0.16)
    dms = [
        rao.Dm(
            alt=rao.Altitude(0.0),
            coupling=pert.dm_coupling,
            microns_per_volt=pert.dm_mpv,
            actu_pos=rao.Positions.rect_grid(21, 8, 8),
            misreg=rao.MisReg((0.0, 0.0), 0.0, 1.0),
        )
    ]
    subap_pos = rao.Positions.rect_grid(20, 7.6, 7.6)

    wfss = [
        rao.Wfs(
            dir=wfs_dirs[0],
            gsalt=rao.Altitude(90e3),
            subap_pos=subap_pos,
            misreg=rao.MisReg(
                (pert.wfs1_delta_x, pert.wfs1_delta_y),
                pert.wfs1_clocking,
                pert.wfs1_zoom,
            ),
        ),
        rao.Wfs(
            dir=wfs_dirs[1],
            gsalt=rao.Altitude(90e3),
            subap_pos=subap_pos,
            misreg=rao.MisReg(
                (pert.wfs2_delta_x, pert.wfs2_delta_y),
                pert.wfs2_clocking,
                pert.wfs2_zoom,
            ),
        ),
        rao.Wfs(
            dir=wfs_dirs[2],
            gsalt=rao.Altitude(90e3),
            subap_pos=subap_pos,
            misreg=rao.MisReg(
                (pert.wfs3_delta_x, pert.wfs3_delta_y),
                pert.wfs3_clocking,
                pert.wfs3_zoom,
            ),
        ),
        rao.Wfs(
            dir=wfs_dirs[3],
            gsalt=rao.Altitude(90e3),
            subap_pos=subap_pos,
            misreg=rao.MisReg(
                (pert.wfs4_delta_x, pert.wfs4_delta_y),
                pert.wfs4_clocking,
                pert.wfs4_zoom,
            ),
        ),
    ]
    ctrl = rao.Ctrl(
        opt_dirs=[(0.0, 0.0)],
        pos=rao.Positions.rect_grid(41, 8, 8),
        dt=0.0,
    )
    atmos = rao.Atmos(layers=[rao.TurbLayer(0.1, 60.0, rao.Altitude(0.0), 10.0, 0.0)])
    system = rao.CompactSystem(telescope, dms, wfss, ctrl, atmos).expand()
    reorder_idx = list(np.array([
        np.arange(2*(20**2)).reshape((2, 20**2)).T.flatten() + i * 2*(20**2)
        for i in range(4)
    ]).flatten().astype(int))
    system.reorder_meas(reorder_idx)
    system.filter_meas(keck_measurement_mask())
    system.filter_com(keck_actuator_mask())
    return system


def build_imat(system: rao.ExpandedSystem) -> NDArray:
    _imat, shape = system.d_meas_com()
    imat = np.array(_imat).reshape(shape)
    return imat


def keck_actuator_mask() -> Sequence[bool]:
    yy, xx = np.meshgrid(np.arange(21), np.arange(21), indexing="ij")
    rr = ((xx.flatten() - xx.mean()) ** 2 + (yy.flatten() - yy.mean()) ** 2) ** 0.5
    r = 10.5
    valid = rr < r
    # for row in valid:
    #     for entry in row:
    #         print("a " if entry else "- ", end="")
    #     print("")
    # print("")
    return list(valid.astype(bool))


def keck_measurement_mask() -> Sequence[bool]:
    yy, xx = np.meshgrid(np.arange(20), np.arange(20), indexing="ij")
    rr = ((xx.flatten() - xx.mean()) ** 2 + (yy.flatten() - yy.mean()) ** 2) ** 0.5
    r = 10
    cobs = 2
    valid = rr < r
    valid &= rr > cobs
    # for row in valid:
    #     for entry in row:
    #         print("s " if entry else "- ", end="")
    #     print("")
    # print("")
    valid = np.tile(np.tile(valid[:, None], [1, 2]).flatten(),[4])
    return list(valid.astype(bool))

if __name__ == "__main__":
    system = kapa(Perturbations())
    imat = build_imat(system)
    print(imat.shape)
    import matplotlib.pyplot as plt
    plt.matshow(imat)
    plt.savefig("tmp.png") 
    