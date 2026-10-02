# Test data

Copied unchanged from [xraylarch](https://github.com/xraypy/xraylarch) 2026.3.1
(MIT License, Copyright (c) 2010-2025 Matthew Newville and the xraylarch authors):

- `cu_metal_rt.xdi` and `fe_athena.prj` from `examples/xafsdata/`
- `feffcu01.dat` from `examples/feffit/`

`fe_fluorescence.xdi` is a synthetic Fe K-edge fluorescence scan (`Fe_Ka / i0`)
with self-absorption damping, for the `larch_fluo_corr` test.

The FEFFIT test in `tests/test_tools.py` reproduces larch's
`examples/feffit/doc_feffit1.py`, whose published result (`doc_feffit1.out`) is
S0^2 0.931(39), sigma^2 0.00867(31), deltaR -0.0060(26) A, E0 4.36(51) eV.
