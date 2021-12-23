# lammps-polymlp-package
A user package of LAMMPS software enabling simulations using polynomial machine learning potentials

Building lammps with lammps-polymlp package
----------------------------------------------
(**lammps-polymlp-package** is tested using LAMMPS_VERSION "3 Mar 2020”)

1. Copy all the components in the **lammps-polymlp-package** to the latest lammps source code directory as
```
    > cp -r lammps-polymlp-package/lib/polymlp $(lammps_src)/lib
    > cp -r lammps-polymlp-package/src/USER-POLYMLP $(lammps_src)/src
```
2. Modify $(lammps_src)/lib/polymlp/Makefile.lammps to specify an installed directory of the boost library.

3. Add "user-polymlp" to variable PACKUSER defined in $(lammps_src)/src/Makefile and activate user-polymlp package as
```
    > vi $(lammps_src)/src/Makefile
        PACKUSER = user-atc user-awpmd user-cgdna user-cgsdk user-colvars \
            user-diffraction user-dpd user-drude user-eff user-fep user-h5md \
            user-intel user-lb user-manifold user-meamc user-mgpt user-misc user-molfile \
            user-netcdf user-omp user-phonon user-qmmm user-qtb \
            user-quip user-reaxc user-smd user-smtbq user-sph user-tally \
            user-vtk user-polymlp

    > make yes-user-polymlp
```
4. Build lammps binary files. (It requires approximately ten minutes for compiling polymlp_gtinv_data.cpp.)
```
    > make serial -j 36
```

Machine learning potentials for a wide range of systems can be found in the website. If you use **lammps-polymlp** package and machine learning potentials in the repository for academic purposes, please cite the following article [1].

[1] A. Seko, A. Togo and I. Tanaka, "Group-theoretical high-order rotational invariants for structural representations: Application to linearized machine learning interatomic potential", Phys. Rev. B 99, 214108 (2019).

Lammps input commands to specify a machine learning potential
------------------------------------------------------------------

The following lammps input commands specify a machine learning potential.
```
    pair_style  polymlp
    pair_coeff * * mlp.lammp Ti Al    
```

