# lammps-mlip-package
A user package of LAMMPS software enabling simulations using linearized machine learning potentials

Building lammps with lammps-mlip package
----------------------------------------------

1. Copy all the components in the **lammps-mlip** package to the latest lammps source code directory as
```
    > cp -r lammps-mlip/lib/mlip $(lammps_src)/lib
    > cp -r lammps-mlip/src/USER-MLIP $(lammps_src)/src
```
2. Modify $(lammps_src)/lib/mlip/Makefile.lammps to specify an installed directory of the boost library.

3. Add "user-mlip" to variable PACKUSER defined in $(lammps_src)/src/Makefile and activate user-mlip package as
``````
    > vi $(lammps_src)/src/Makefile
        PACKUSER = user-atc user-awpmd user-cgdna user-cgsdk user-colvars \
            user-diffraction user-dpd user-drude user-eff user-fep user-h5md \
            user-intel user-lb user-manifold user-meamc user-mgpt user-misc user-molfile \
            user-netcdf user-omp user-phonon user-qmmm user-qtb \
            user-quip user-reaxc user-smd user-smtbq user-sph user-tally \
            user-vtk user-mlip

    > make yes-user-mlip
```
4. Build lammps binary files ::
```
    > make serial -j 36
```
