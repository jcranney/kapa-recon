from matplotlib.axes import Axes
import pyrao as rao
import numpy as np
from typing import List, Sequence, Tuple
from numpy.typing import NDArray
from dataclasses import dataclass, fields
import json

TEL_DIAM: float = 10.0
COBS: float = 0.16
NSUBX: int = 20
NACTX: int = 21
GS_ALT: float = 90e3
NSUBAP_SAMPLES: int = 3
AS2RAD: float = np.pi / 180 / 60 / 60
SPIDER_THICKNESS: float = 0.0


@dataclass
class Perturbations:
    dm_coupling: float = 0.1
    dm_mpv: float = 0.05  # microns per volt
    dm_aspect_ratio: float = 1.0
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
    wfs_dirs: List[Tuple[float, float]] = [
        (+5.5, +5.5),
        (+5.5, -5.5),
        (-5.5, -5.5),
        (-5.5, +5.5),
    ]
    wfs_dirs = [(d[0] * AS2RAD, d[1] * AS2RAD) for d in wfs_dirs]
    telescope = rao.Telescope(teldiam=TEL_DIAM, cobs=0.0)
    dms = [
        rao.Dm(
            alt=rao.Altitude(0.0),
            coupling=pert.dm_coupling,
            microns_per_volt=pert.dm_mpv,
            actu_pos=rao.Positions.rect_grid(NACTX, TEL_DIAM*pert.dm_aspect_ratio, TEL_DIAM),
            misreg=rao.MisReg(delta=(0.0, 0.0), clocking=0.0, zoom=1.0),
        )
    ]
    subap_pos = rao.Positions.rect_grid(
        NSUBX, TEL_DIAM * (NSUBX - 1) / NSUBX, TEL_DIAM * (NSUBX - 1) / NSUBX
    )
    wfss = [
        rao.Wfs(
            dir=wfs_dirs[0],
            gsalt=rao.Altitude(GS_ALT),
            subap_pos=subap_pos,
            misreg=rao.MisReg(
                (pert.wfs1_delta_x, pert.wfs1_delta_y),
                pert.wfs1_clocking,
                pert.wfs1_zoom,
            ),
            subap_samples=NSUBAP_SAMPLES,
            pupil=rao.Pupil(TEL_DIAM, COBS, spiders=rao.Spiders([], SPIDER_THICKNESS))
        ),
        rao.Wfs(
            dir=wfs_dirs[1],
            gsalt=rao.Altitude(GS_ALT),
            subap_pos=subap_pos,
            misreg=rao.MisReg(
                (pert.wfs2_delta_x, pert.wfs2_delta_y),
                pert.wfs2_clocking,
                pert.wfs2_zoom,
            ),
            subap_samples=NSUBAP_SAMPLES,
            pupil=rao.Pupil(TEL_DIAM, COBS, spiders=rao.Spiders([], SPIDER_THICKNESS))
        ),
        rao.Wfs(
            dir=wfs_dirs[2],
            gsalt=rao.Altitude(GS_ALT),
            subap_pos=subap_pos,
            misreg=rao.MisReg(
                (pert.wfs3_delta_x, pert.wfs3_delta_y),
                pert.wfs3_clocking,
                pert.wfs3_zoom,
            ),
            subap_samples=NSUBAP_SAMPLES,
            pupil=rao.Pupil(TEL_DIAM, COBS, spiders=rao.Spiders([], SPIDER_THICKNESS))
        ),
        rao.Wfs(
            dir=wfs_dirs[3],
            gsalt=rao.Altitude(GS_ALT),
            subap_pos=subap_pos,
            misreg=rao.MisReg(
                (pert.wfs4_delta_x, pert.wfs4_delta_y),
                pert.wfs4_clocking,
                pert.wfs4_zoom,
            ),
            subap_samples=NSUBAP_SAMPLES,
            pupil=rao.Pupil(TEL_DIAM, COBS, spiders=rao.Spiders([], SPIDER_THICKNESS))
        ),
    ]
    ctrl = rao.Ctrl(
        opt_dirs=[(0.0, 0.0)],
        pos=rao.Positions.rect_grid(41, TEL_DIAM, TEL_DIAM),
        dt=0.0,
    )
    # TODO: add cn2 profile from .pro file here
    atmos = rao.Atmos(layers=[rao.TurbLayer(0.1, 60.0, rao.Altitude(0.0), 10.0, 0.0)])
    system = rao.CompactSystem(telescope, dms, wfss, ctrl, atmos).expand()
    reorder_idx = list(
        np.array(
            [
                np.arange(2 * (NSUBX**2)).reshape((2, NSUBX**2)).T.flatten()
                + i * 2 * (NSUBX**2)
                for i in range(4)
            ]
        )
        .flatten()
        .astype(int)
    )
    system.reorder_meas(reorder_idx)
    system.filter_meas(keck_measurement_mask())
    system.filter_com(keck_actuator_mask())
    return system


def build_imat(system: rao.ExpandedSystem) -> NDArray:
    _imat, shape = system.d_meas_com()
    imat = np.array(_imat).reshape(shape)
    return imat


def keck_actuator_mask() -> Sequence[bool]:
    yy, xx = np.meshgrid(np.arange(NACTX), np.arange(NACTX), indexing="ij")
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
    yy, xx = np.meshgrid(np.arange(NSUBX), np.arange(NSUBX), indexing="ij")
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
    valid = np.tile(np.tile(valid[:, None], [1, 2]).flatten(), [4])
    return list(valid.astype(bool))


def plot_system(system: rao.ExpandedSystem, ax: Axes):
    # fig = plt.figure(figsize=[10, 10])
    altitude = 0.0
    meas_coords = system.meas_coords(altitude)
    ax.axvline(color="k", linestyle=":", linewidth=0.5)
    ax.axhline(color="k", linestyle=":", linewidth=0.5)
    legend = {}
    for w in range(4):
        # for each wfs
        color = [
            "#880088",
            "#888800",
            "#008888",
            "#448844",
        ][w]
        for sa in range(0, 608, 2):
            # for each subaperture
            subap_index: int = w * 304 * 2 + sa
            corners = meas_coords[subap_index].corners()
            corners += [corners[0]]
            corners_arr = np.array(corners).T
            # plot a square with the appropriate position
            (line,) = ax.plot(*corners_arr, color=color, label=f"WFS-{w+1:d}")
            legend[f"WFS-{w+1}"] = line

    actu_coords = np.array([x.pos() for x in system.actu_coords()])
    (line,) = ax.plot(*actu_coords.T, "x", color="#ff0000", label="DM")
    legend["DM"] = line
    ax.legend(handles=legend.values())
    ax.set_title(f"WFS/DM Registration, KAPA, {altitude:0.1f} km")
    ax.axis("square")


if __name__ == "__main__":
    system = kapa(Perturbations())
    imat = build_imat(system)
    print(imat.shape)
    import matplotlib.pyplot as plt

    plt.matshow(imat)
    plt.savefig("tmp.png")
