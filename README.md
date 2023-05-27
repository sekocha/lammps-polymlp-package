# lammps-polymlp-package
A package of LAMMPS software enabling simulations using polynomial machine learning potentials

Building lammps with lammps-polymlp package
----------------------------------------------
(**lammps-polymlp-package** is tested using LAMMPS_VERSION "lammps-23Jun2022”)

1. Copy all the components in the **lammps-polymlp-package** to the latest lammps source code directory as
```
    > cp -r lammps-polymlp-package/lib/polymlp $(lammps_src)/lib
    > cp -r lammps-polymlp-package/src/POLYMLP $(lammps_src)/src
```

2. Add "polymlp" to variable PACKAGE defined in $(lammps_src)/src/Makefile and activate polymlp package as
```
    > cat $(lammps_src)/src/Makefile
        PACKAGE = \
        adios \
        amoeba \
        ...
        poems \
        polymlp \
        ptm \
        ...
        ml-iap \
        phonon
        ...
    > ulimit -s unlimited
    > cd $(lammps_src)/src
    > make yes-user-polymlp
```
3. Build lammps binary files. (It requires approximately ten minutes to one hour for compiling polymlp_gtinv_data.cpp.)
```
    > make serial -j 36
```

Machine learning potentials for a wide range of systems can be found in the website. If you use **lammps-polymlp** package and machine learning potentials in the repository for academic purposes, please cite the following article [1].

[1] A. Seko, "Systematic development of polynomial machine learning potentials for elemental and alloy systems", J. Appl. Phys. 133, 011101 (2023).

Lammps input commands to specify a machine learning potential
------------------------------------------------------------------

The following lammps input commands specify a machine learning potential.
```
    pair_style  polymlp
    pair_coeff * * mlp.lammp Ti Al    
```

