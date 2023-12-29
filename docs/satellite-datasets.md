Satellite Datasets
==================

Satellite datasets should always contain files belonging to one specific satellite,
instrument and wavelength only.

This document describes the expected structure for supported satellite datasets which
are required for passing the validation checks and working with the included patch
generation script. Datasets downloaded/created using the provided downloader script
should follow these structures automatically. All datasets are intentionally structured
to mirror their corresponding online archives to make it easier to find the online URL
for a locally downloaded file.

Notes
-----
### Dataset Names
Dataset names are arbitrary but by convention follow the pattern
*{satellite}-{instrument}-{wavelength}* with all lowercase letters, e.g. *vco-uvi-365*
or *vex-vmc-uv*.

### Geometry Files
Both Akatsuki and Venus Express / VMC datasets consist of corresponding pairs of image
and geometry files that can be linked together via their file names (and also via their
relative paths inside the dataset directories). It is possible that an image file in the
online archive does not have a corresponding geometry file. In that case it must be
removed from the local dataset.

### 'X' Symbols and '...'
An `X` in a directory/file name represents an arbitrary digit from 0 to 9. Usually
such directories/files are followed by a '...', indicating that they are followed by
one or multiple (but possibly also 0) directories/files with different numbers and
identical substructures.

### Brace Fields
Text enclosed in braces reprents a string that is consistent throughout a particular
dataset directory but varies between datasets of a satellite. Their possible values are
listed below the directory structures.

Akatsuki (VCO) Datasets
-----------------------
### Directory Structure
```sh
dataset-name/
├── extras/
│   ├── vco_{instrument}_l3_vX.X/  # Geometry file directories, e.g. 'vco_uvi_l3_v1.0'
│   │   ├── vco{instrument}_7XXX/  # Mission extension directories, e.g. 'vcouvi_7001'
│   │   │   └── data/l3bx/fits/
│   │   │       ├── c0000/  # Cruise phase directory (same substructure as orbit directories)
│   │   │       ├── rXXXX/  # Orbit directories, e.g. 'r0001'
│   │   │       │   ├── {instrument}_YYYYMMDD_hhmmss_{wavelength}_l3bx_vXX.fit  # Geometry files, e.g. 'uvi_20151207_172945_365_l3bx_v10.fit'
│   │   │       │   └── ...
│   │   │       └── ...
│   │   └── ...
│   └── ...
├── vco-v-{instrument}-3-cdr-vX.X/  # Image file directories, e.g. 'vco-v-uvi-3-cdr-v1.0'
│   ├── vco{instrument}_1XXX/  # Mission extension directories, e.g. 'vcouvi_1001'
│   │   └── data/l2b/
│   │       ├── c0000/  # Cruise phase directory (same substructure as orbit directories)
│   │       ├── rXXXX/  # Orbit directories, e.g. 'r0001'
│   │       │   ├── {instrument}_YYYYMMDD_hhmmss_{wavelength}_l2b_vXX.fit  # Image files, e.g. 'uvi_20151207_172945_365_l2b_v10.fit'
│   │       │   └── ...
│   │       └── ...
│   └── ...
└── ...
```

### Possible Brace Field Values
- `{instrument}`: `ir1`, `ir2`, `lir`, `uvi`
- `{wavelength}`: `097`, `09d`, `09n`, `101`, `165`, `174`, `202`, `226`, `232`, `pic`, `283`, `365`

### File Versions
Akatsuki dataset files can exist in multiple versions (noted at the end of the file
name, e.g. 'v10'). A local dataset must only contain one version for each file (ideally
the highest version).

Venus Express (VEX) / Venus Monitoring Camera (VMC) Datasets
------------------------------------------------------------
### Directory Structure
```sh
dataset-name/
├── VEX-V-VMC-3-RDR-V3.0/  # Original mission directory (same substructure as mission extension directories)
├── VEX-V-VMC-3-RDR-EXTX-VX.X/  # Mission extension directories, e.g. 'VEX-V-VMC-3-RDR-EXT1-V3.0'
│   ├── DATA/
│   │   ├── XXXX/  # Orbit directories, e.g. '0550'
│   │   │   ├── VXXXX_XXXX_{wavelength}2.IMG  # Image file, e.g. 'V0550_0001_UV2.IMG'
│   │   │   └── ...
│   │   └── ...
│   └── GEOMETRY/
│       ├── XXXX/  # Orbit directories, e.g. '0550'
│       │   ├── VXXXX_XXXX_{wavelength}2.GEO  # Geometry file, e.g. 'V0550_0001_UV2.GEO'
│       │   └── ...
│       └── ...
└── ...
```

### Possible Brace Field Values
- `{wavelength}`: `N1`, `N2`, `UV`, `VI`
