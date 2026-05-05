import numpy as np
import matplotlib.pyplot as plt

def load_measured_recon(filename: str = "./input_data/16Apr0005.mr"):
    with open(filename,"rb") as f:
        data = f.read()
    data_array = np.frombuffer(data, dtype=np.dtype("float32").newbyteorder(">"))
    data_array = data_array.reshape([360, 2432])
    return data_array

if __name__ == "__main__":
    data_array = load_measured_recon()
    plt.figure(figsize=[12, 4])
    plt.imshow(data_array)
    plt.colorbar()
    plt.xlabel("measurements")
    plt.ylabel("actuators")
    plt.tight_layout()
    plt.savefig("./output_data/measured_recon.png", dpi=300)