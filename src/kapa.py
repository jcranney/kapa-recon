import pyrao  # type: ignore
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass, fields
import json


NWFS: int = 1
NDM: int = 1
NPARAM_DM = 6
NPARAM_WFS = 4

@dataclass
class Perturbations:
    dm_coupling: float = 0.3
    dm_mpv: float = 1.0  # microns per volt
    wfs_delta_x: float = 0.0
    wfs_delta_y: float = 0.0
    wfs_clocking: float = 0.0
    wfs_zoom: float = 0.0

    @staticmethod
    def from_list(pert_list: List[float]):
        kwargs = {}
        for i, field in enumerate(fields(Perturbations)):
            kwargs[field.name] = pert_list[i]
        return Perturbations(
            **kwargs
        )

    def to_list(self) -> List[float]:
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


def build_imat(perturbations: Perturbations, lgs_dir: Tuple[float, float], indices=None):
    """
    take the perturbation vector and generate either a full of sparse imat
    """
    
    system_geom = build_system(perturbations, lgs_dir)

    # sampling the imat is typically the expensive part:
    if indices is None:
        # the first time through, we build the nominal imat
        # takes ~20 seconds on my laptop
        imat = np.array(system_geom.imat())
    else:
        # every other time, we only sample it sparsely
        # takes ~2ms per 1000 samples
        imat = np.array(system_geom.imat_sparse(indices))
    return imat


def build_system(perturbations: Perturbations, lgs_dir: Tuple[float, float]):
    """
    take the perturbation vector and generate the keck system
    geometry for a single WFS
    """
    # building the entire system is very quick, not much computation required
    system_geoms = []
    system_geoms.append(
        pyrao.SystemGeom.new(
            teldiam=8.0,
            cobs=0.16,
            coupling=perturbations.dm_coupling,
            nactux=21,
            dmalt=0.0,
            pitch=0.40,
            nsubx=20,
            ntssamples=41,
            nphisamples=41,
            wfs_dirs=[lgs_dir],
            ts_dirs=[(0.0, 0.0)],
            dm_delta=(0.0, 0.0),
            wfs_delta=(
                (perturbations.wfs_delta_x, perturbations.wfs_delta_y),
            ),
            dm_clocking=0.0,
            wfs_clocking=(
                perturbations.wfs_clocking,
            ),
            dm_zoom=0.0,
            wfs_zoom=(
                perturbations.wfs_zoom,
            ),
            gsalt=90e3,
            microns_per_volt=perturbations.dm_mpv,
        )
    )
    system_geom = pyrao.SystemGeom.merge_com(system_geoms)
    reorder_idx = np.arange(2*(20**2)).reshape((2, 20**2)).T.flatten()
    system_geom.reorder_meas(reorder_idx)

    valid_actuators = get_keck_actuator_mask()
    system_geom.filter_com(valid_actuators)
    valid_measurements = get_keck_measurement_mask()
    system_geom.filter_meas(valid_measurements)

    return system_geom


def get_keck_actuator_mask() -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(21), np.arange(21), indexing="ij")
    rr = ((xx - xx.mean())**2 + (yy - yy.mean())**2)**0.5
    r = 10.5
    valid = rr < r
    # for row in valid:
    #     for entry in row:
    #         print("a " if entry else "- ", end="")
    #     print("")
    # print("")
    return valid.flatten()


def get_keck_measurement_mask() -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(20), np.arange(20), indexing="ij")
    rr = ((xx - xx.mean())**2 + (yy - yy.mean())**2)**0.5
    r = 10
    cobs = 2
    valid = rr < r
    valid &= rr > cobs
    # for row in valid:
    #     for entry in row:
    #         print("s " if entry else "- ", end="")
    #     print("")
    # print("")
    valid = np.tile(valid.flatten()[:, None], [1, 2]).flatten()
    return valid
