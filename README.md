# kapa-recon

This package aims to generate control matrices (reconstructors) for the KAPA AO
system on Keck. At the core of this project is [pyrao](https://github.com/jcranney/pyrao).

## Getting Started

```bash
git clone git@github.com:jcranney/kapa-recon
cd kapa-recon
uv sync
source .venv/bin/activate
```

If the above completed successfully, you should have a virtual environment
with the `kapa-recon` package available. Then you can run:
```bash
fit-imat --help
```
and 
```bash
build-cmat --help
```

## See also:
 - [kapa imat fitting](./KAPA_imat_fitting.md)