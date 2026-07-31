import numpy as np
import matplotlib.pyplot as plt
import glob


def read_array(filename: str) -> np.ndarray:
    with open(filename, "rb") as f:
        data = f.read()
    dtype = np.dtype(">f4")
    array = np.frombuffer(data, dtype=dtype)
    imat = array.reshape([-1, 349])
    return imat


def write_array(filename: str, array: np.ndarray):
    reshaped = array.flatten().astype(">f4").tobytes()
    with open(filename, "wb") as f:
        f.write(reshaped)


if __name__ == "__main__":
    for filename in glob.glob("24.imx-LGS*"):
        imat = read_array(filename)
        plt.matshow(imat)
        plt.savefig(filename.split("-")[-1]+".png")