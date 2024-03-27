Satellite Datasets
==================

Satellite datasets always contain files belonging to one specific satellite, instrument
and wavelength only.

This document describes where to download the required files and the
expected file/directory structures for the supported satellite datasets which are
required for passing the validation checks and working with the included patch
generation script. All datasets are intentionally structured to mirror their
corresponding online archives to make it easier to find the online URL for a locally
downloaded file.

Downloads
---------
### Akatsuki / Venus Climate Orbiter (*VCO*)
The files for these datasets can be downloaded from this archive:
https://darts.isas.jaxa.jp/planet/project/akatsuki/.

Data from the *UVI*, *IR1*, *IR2*, and *LIR* instruments should be compatible with the
provided code/scripts. In each case you should download the *Level 2 (b)* calibrated
image *FITS* (`.fit`) files and the correspndong *Level 3 (bx)* pointing corrected
geometry (see [below](#geometry-files)) *FITS* files. You can simply download and unpack
the zip files. As mentioned above, you should seperate the files corresponding to
different instruments and wavelengths and structure the directory tree according to
[this](#akatsuki-vco-datasets). The datasets should then be placed at the location
described [below](#dataset-paths).

### Venus Express (*VEX*) / Venus Monitoring Camera (*VMC*)
The image and geometry files for these datasets can be downloaded from the *VMC FTP*
directory at this archive: https://www.cosmos.esa.int/web/psa/venus-express.

You should download both the image `.IMG` and `.GEO` files in the respective `DATA` and
`GEOMETRY` directories. You can only download the wavelengths (e.g. `UV2`) you need.
The files need to be structured according to
[this](#venus-express-vex--venus-monitoring-camera-vmc-datasets).

You also need to download the
[*SPICE* kernels](https://naif.jpl.nasa.gov/naif/data.html) from:
https://s2e2.cosmos.esa.int/bitbucket/projects/SPICE_KERNELS/repos/venus-express/browse/kernels.

The datasets and *SPICE* kernels should then placed at the location described
[below](#dataset-paths).

### Juno (*JNO*) / JunoCam (*JNC*) 
The image files for this dataset can be downloaded from the this archive:
https://planetarydata.jpl.nasa.gov/img/data/juno/.

Here you simply need to download the `JNOJNC_XXXX.tar.gz` zip files. Once unpacked,
the important files will be located in the `DATA/RDR/` subdirectories. The files need to
be structured according to [this](#juno-jnc--junocam-jnc-datasets).

You also need to download the
[*SPICE* kernels](https://naif.jpl.nasa.gov/naif/data.html) from:
https://naif.jpl.nasa.gov/pub/naif/pds/data/jno-j_e_ss-spice-6-v1.0/jnosp_1000/data/.

The datasets and *SPICE* kernels should then placed at the location described
[below](#dataset-paths).

Notes
-----
### Dataset Paths
The dataset directories should be placed at
`<DATA_DIR_PATH>/data/satellite-datasets/<planet>/archives/<archive name>/datasets/`,
where `<planet>` must be either `venus` or `jupiter` and `<archive name>` must be
`vco` for *Akatsuki* datasets, `vex-vmc` for *Venus Express VMC* datasets, and `jno-jnc`
for the *Juno JunoCam* dataset. `<DATA_DIR_PATH>` is a custom path set in the
`config.py` file (see *Setup* step 2 in [README.md](../README.md#setup)).

In the `user/resources/` directory, you will find two files named `planet-jupiter.json`
and `planet-venus.json`. You need to copy these files, place them in their respective
`<DATA_DIR_PATH>/data/satellite-datasets/<planet>/` subdirectories, and rename them to
just `planet.json`.

The downloaded *SPICE* kernel directories (`ik/`, `ck/` etc.) need to be placed into
`<DATA_DIR_PATH>/data/satellite-datasets/<planet>/archives/<archive name>/spice-kernels/`
for their corresponding archive.

### Dataset Names
Dataset names (i.e. names of dataset directories) are arbitrary but should follow the
pattern *{satellite}-{instrument}-{wavelength}* with all lowercase letters, e.g.
*vco-uvi-365* or *vex-vmc-uv*.

### Geometry Files
Both the Akatsuki and Venus Express / VMC datasets consist of corresponding pairs of
image and geometry files that can be linked together via their file names (and also via
their relative paths inside the dataset directories). It is possible that an image file
in the online archive does not have a corresponding geometry file. In that case it must
be removed from the local dataset.

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
<dataset name>/data/
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
<dataset name>/data/
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


Juno (JNO) / JunoCam (JNC) Dataset
----------------------------------
```sh
<dataset name>/data/
├── JNOJNC_XXXX/  # Volume directories, e.g. 'JNOJNC_0001'
│   └── RDR/JUPITER/
│       ├── ORBIT_XX  # Orbit directories, e.g. 'ORBIT_39'
│       │   ├── JNCR_YYYYDDD_OOFNNNNN_VXX.IMG  # Image file, e.g. 'JNCR_2016129_00C00158_V01.IMG'
│       │   ├── JNCR_YYYYDDD_OOFNNNNN_VXX.LBL  # PDS3 label file, e.g. 'JNCR_2016129_00C00158_V01.LBL'
│       │   └── ...
│       └── ...
└── ...
```


Cloud-ImVN 1.0 Patch Dataset
============================
A labeled Earth cloud image dataset called *Cloud-ImVN 1.0` and that acts similar to the
generated patch datasets can be downloaded at: https://data.mendeley.com/datasets/vwdd9grvdp/2.

Once downloaded, merge all images into a single directory (and name it e.g. `default`)
and place it at `<DATA_DIR_PATH>/patch-datasets/cloud-imvn-1.0/cloud-imvn-1.0/versions/`.
Then copy the `user/resources/cloud-imvn-1.0_info.json` file, place it into
`<DATA_DIR_PATH>/patch-datasets/cloud-imvn-1.0/cloud-imvn-1.0/`, and rename it to just
`info.json`. This dataset should now be usable with all patch-related scripts.