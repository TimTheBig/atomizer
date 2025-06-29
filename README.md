# Atomizer: Beyond Non-Planar Slicing for Fused Filament Fabrication

![Representative image of the repository](data/image/doc/representative_image.png)

This repository contains the implementation of **Atomizer**, a novel toolpath generation algorithm for fused filament fabrication (FFF) 3D printing. Atomizer introduces the concept of *atoms* as a replacement for traditional slices. This method enables collision-free toolpaths that tightly conform to complex geometries and offer new fabrication capabilities.

- [Project page](https://xavierchermain.github.io/publications/atomizer)
- [Article](https://drive.google.com/file/d/1P4MiYd7Qz1rJdqrq3-CEuzC4vQ2PIMeO/view?usp=sharing)
- [Supplemental Video](https://youtu.be/d9SBcfkywqA)

*[Xavier Chermain](https://xavierchermain.github.io), [Giovanni Cocco](https://github.com/iota97), [Cédric Zanni](https://members.loria.fr/CZanni/), Eric Garner, [Pierre-Alexandre Hugron](https://www.linkedin.com/in/pierre-alexandre-hugron-b7b22552), and [Sylvain Lefebvre](https://www.antexel.com/sylefeb/research)*

[Université de Lorraine](https://www.univ-lorraine.fr/en/univ-lorraine/), [CNRS](https://www.cnrs.fr/en), [Inria](https://www.inria.fr/en), [Loria](https://www.loria.fr/en/)

[Computer Graphics Forum](https://doi.org/10.1111/cgf.70189) ([Proceedings of the Symposium on Geometry Processing](https://sgp2025.my.canva.site/program-page-sgp)), 2025

## Key Features

* Toolpath generation using frames (i.e., atoms) instead of slices
* Control of the deposition direction and tool orientation
* Collision-free fabrication ordering
* Support for anisotropic appearance on curved surfaces

## Installation

### Clone the Repository

To clone the code
```
git clone https://github.com/xavierchermain/atomizer.git
```

### Install Python Module Dependencies

This implementation is written in Python 3.10 and uses [Taichi](https://www.taichi-lang.org/). The requirements are

* Python >=3.7 and <3.11
* Taichi
* NumPy
* Pyvista
* tqdm
* Pillow
* Blender
* the local library [`src/atom`](src/atom).

You can use `pip` to install the python dependencies. The following commands should be run from the repository root directory:
```
pip install --user -e .
```

#### Conda

For Conda users, you can try
```
conda create --name atomizer python=3.10
conda activate atomizer
conda install pip
pip install --user -e .
```

#### Blender

Blender is only used to automatically remesh the 3D model for the atomizer. You need to install it and add it to the executable path.

## Repository Structure

```
atomizer/
├── tools/                             # The main tools
|    └── atomize.py                    # Main pipeline entry point
|    └── bpn_to_sdf.py                 # SDF generation and voxelization
|    └── compute_tool_orientations.py  # Tool orientation field optimization
|    └── compute_tangents.py           # Deposition tangent field optimization
|    └── align_atoms.py                # Tricosine field optimization
|    └── extract_explicit_atoms.py     # Atom extraction
|    └── order_atoms.py                # Atom ordering
|    └── visualize_bpn_sdf.py          # Visualize SDF
|    └── visualize_bpn_df.py           # Visualize tool orientation field
|    └── visualize_bases.py            # Visualize deposition tangent field
|    └── visualize_implicit_atoms.py   # Visualize tricosine field
|    └── visualize_explicit_atoms.py   # Visualize explicit atoms
|    └── visualize_toolpath.py         # Visualize generated toolpath
|    └── ...                           # Other tools
├── data/                              # Input and output data
|    └── mesh/                         # Input STLs
|    └── param/                        # Input parameter JSON files
|    └── toolpath/                     # Output toolpaths
|    └── ...                           # Other data
├── src/atom/                          # Local library with core functionalities
├── experiment/                        # To experiment some core functionalities
├── generate_results.ps1               # Script generating the toolpaths
└── README.md
```

## Usage

```
python tools/atomize.py data/param/triangle_24.json
```

JSON example:
```
{
    "solid_name": "triangle_24",
    "deposition_width": 0.9,
    "max_slope": 5.5
}
```

With the previous example, the input and outputs are
- STL input: data/mesh/triangle_24.stl
- Toolpath output: data/toolpath/triangle_24.npz
- Log file: data/log/triangle_24.log

Look at the log file to see all the individual computation and visualization
commands for the model.

Add the `--warmup` option to exclude the compilation time in the computation
time in the log file. Caution: this runs two times each step of the pipeline as
Taichi is a just-in-time compiler.

### Commands manuals

Tools' manuals are accessible by typing `--help`, e.g.,
```
python tools/atomize.py --help
```

## 3D Printer

Atomizer was tested on a custom 3-axis printer with independently controlled Z-axis screws. The custumization is for experts, and consequently, we do not provide any GCODE to prevent people damaging their machine. The code only generates the sequence of positions, each associated with a tool orientation and a travel type (deposition or no deposition). To print, you have to use your machine inverse kinematics.

## Citation

If you use this code in your research, please cite:

```
@article{Chermain2025Atomizer,
author = {Chermain, Xavier and Cocco, Giovanni and Zanni, Cédric and Garner, Eric and Hugron, Pierre-Alexandre and Lefebvre, Sylvain},
title = {{Atomizer: Beyond Non-Planar Slicing for Fused Filament Fabrication}},
journal = {Computer Graphics Forum (Proceedings of the Symposium on Geometry Processing)},
year = {2025}
doi = {https://doi.org/10.1111/cgf.70189},
}
```

## License

The source code is under the BSD 3-Clause "New" or "Rivised" license. See
[LICENSE](LICENSE) for more details.
