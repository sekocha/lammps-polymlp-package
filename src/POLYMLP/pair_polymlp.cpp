/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Atsuto Seko
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "atom.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "force.h"
#include "comm.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"

#include "pair_polymlp.h"
#include "time.h"

#include <omp.h>
using namespace LAMMPS_NS;

#define MAXLINE 1024
#define DELTA 4

/* ---------------------------------------------------------------------- */

PairPolyMLP::PairPolyMLP(LAMMPS *lmp) : Pair(lmp)
{
    restartinfo = 0;
}


/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairPolyMLP::~PairPolyMLP()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

/* ---------------------------------------------------------------------- */

void PairPolyMLP::compute(int eflag, int vflag)
{
    const auto& fp = polymlp.get_fp();
    set_types();
    if (fp.feature_type == "pair"){
        compute_pair(eflag, vflag);
    }
    else if (fp.feature_type == "gtinv"){
        compute_gtinv(eflag, vflag);
    }
}

/* ---------------------------------------------------------------------- */

void PairPolyMLP::compute_pair(int eflag, int vflag)
{
    vflag = 1;
    if (eflag || vflag) ev_setup(eflag,vflag);
    else evflag = 0;

    int inum = list->inum;
    int nlocal = atom->nlocal;
    int newton_pair = force->newton_pair;

    vector2d antp, prod_sum_e, prod_sum_f;
    compute_antp(antp);
    compute_sum_of_prod_antp(antp, prod_sum_e, prod_sum_f);

    vector2d evdwl_array(inum), fpair_array(inum);
    const auto& fp = polymlp.get_fp();
    const auto& maps = polymlp.get_maps();
    const auto& type_pairs = maps.type_pairs;
    const auto& tp_to_params = maps.tp_to_params;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided)
    #endif
    for (int ii = 0; ii < inum; ii++) {
        int i,j,jnum,*jlist,type1,type2,tp,tagi,tagj;
        double delx,dely,delz,dis,evdwl,fpair;

        double **x = atom->x;
        tagint *tag = atom->tag;

        i = list->ilist[ii];
        tagi = tag[i]-1;
        type1 = types[tagi];
        jnum = list->numneigh[i];
        jlist = list->firstneigh[i];

        evdwl_array[ii].resize(jnum);
        fpair_array[ii].resize(jnum);

        const auto& maps_type = maps.maps_type[type1];
        const auto& ntp_attrs = maps_type.ntp_attrs;

        vector1d fn,fn_d;
        for (int jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            tagj = tag[j]-1;
            type2 = types[tagj];
            delx = x[i][0]-x[j][0];
            dely = x[i][1]-x[j][1];
            delz = x[i][2]-x[j][2];
            dis = sqrt(delx*delx + dely*dely + delz*delz);
            if (dis < fp.cutoff){
                tp = type_pairs[type1][type2];
                const auto& params = tp_to_params[tp];
                get_fn_(dis, fp, params, fn, fn_d);
                evdwl = 0.0, fpair = 0.0;
                for (const auto& ntp: ntp_attrs){
                    if (tp == ntp.tp){
                        const int idx_i = ntp.ilocal_id;
                        const int idx_j = ntp.jlocal_id;
                        const auto& prod_ei = prod_sum_e[tagi][idx_i];
                        const auto& prod_ej = prod_sum_e[tagj][idx_j];
                        const auto& prod_fi = prod_sum_f[tagi][idx_i];
                        const auto& prod_fj = prod_sum_f[tagj][idx_j];
                        evdwl += fn[ntp.n_id] * (prod_ei + prod_ej);
                        fpair += fn_d[ntp.n_id] * (prod_fi + prod_fj);
                    }
                }
                fpair *= - 1.0 / dis;
                evdwl_array[ii][jj] = evdwl;
                fpair_array[ii][jj] = fpair;
            }
        }
    }

    int i,j,jnum,*jlist;
    double fpair,evdwl,dis,delx,dely,delz;
    double **f = atom->f;
    double **x = atom->x;
    for (int ii = 0; ii < inum; ii++) {
        i = list->ilist[ii];
        jnum = list->numneigh[i], jlist = list->firstneigh[i];
        for (int jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            delx = x[i][0]-x[j][0];
            dely = x[i][1]-x[j][1];
            delz = x[i][2]-x[j][2];
            dis = sqrt(delx*delx + dely*dely + delz*delz);
            if (dis < fp.cutoff){
                evdwl = evdwl_array[ii][jj];
                fpair = fpair_array[ii][jj];
                f[i][0] += fpair*delx;
                f[i][1] += fpair*dely;
                f[i][2] += fpair*delz;
                // if (newton_pair || j < nlocal)
                f[j][0] -= fpair*delx;
                f[j][1] -= fpair*dely;
                f[j][2] -= fpair*delz;
                if (evflag) {
                    ev_tally(i,j,nlocal,newton_pair,
                            evdwl,0.0,fpair,delx,dely,delz);
                }
            }
        }
    }
}

void PairPolyMLP::compute_antp(vector2d& antp){

    const auto& fp = polymlp.get_fp();
    const auto& maps = polymlp.get_maps();
    const auto& type_pairs = maps.type_pairs;
    const auto& tp_to_params = maps.tp_to_params;

    int inum = list->inum;
    antp = vector2d(inum);
    for (int ii = 0; ii < inum; ii++){
        tagint *tag = atom->tag;
        int i = list->ilist[ii];
        int type1 = types[tag[i]-1];

        const auto& maps_type = maps.maps_type[type1];
        const auto& ntp_attrs = maps_type.ntp_attrs;
        antp[tag[i]-1] = vector1d(ntp_attrs.size(), 0.0);
    }

    #ifdef _OPENMP
    #pragma omp parallel for schedule(auto)
    #endif
    for (int ii = 0; ii < inum; ii++) {
        int i,j,type1,type2,tp,jnum,*ilist,*jlist;
        double delx,dely,delz,dis;

        double **x = atom->x;
        tagint *tag = atom->tag;

        i = list->ilist[ii];
        type1 = types[tag[i]-1];
        jnum = list->numneigh[i];
        jlist = list->firstneigh[i];

        const auto& maps_type = maps.maps_type[type1];
        const auto& ntp_attrs = maps_type.ntp_attrs;

        vector1d fn; 
        for (int jj = 0; jj < jnum; ++jj) {
            j = jlist[jj];
            delx = x[i][0]-x[j][0];
            dely = x[i][1]-x[j][1];
            delz = x[i][2]-x[j][2];
            dis = sqrt(delx*delx + dely*dely + delz*delz);
            if (dis < fp.cutoff){
                type2 = types[tag[j]-1];
                tp = type_pairs[type1][type2];
                const auto& params = tp_to_params[tp];
                get_fn_(dis, fp, params, fn);
                for (const auto& ntp: ntp_attrs){
                    if (tp == ntp.tp){
                        const int idx_i = ntp.ilocal_id;
                        const int idx_j = ntp.jlocal_id;
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        antp[tag[i]-1][idx_i] += fn[ntp.n_id];
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        antp[tag[j]-1][idx_j] += fn[ntp.n_id];
                    }
                }
            }
        }
    }
}

void PairPolyMLP::compute_sum_of_prod_antp(
    const vector2d& antp, vector2d& prod_sum_e, vector2d& prod_sum_f
){
    const int inum = list->inum;
    prod_sum_e = vector2d(inum);
    prod_sum_f = vector2d(inum);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided)
    #endif
    for (int ii = 0; ii < inum; ii++) {
        tagint *tag = atom->tag;
        const int i = list->ilist[ii];
        const int type1 = types[tag[i]-1];
        polymlp.compute_sum_of_prod_antp(
            antp[tag[i]-1], type1, prod_sum_e[tag[i]-1], prod_sum_f[tag[i]-1]
        );
    }
}

/* ---------------------------------------------------------------------- */

void PairPolyMLP::compute_gtinv(int eflag, int vflag)
{

    vflag = 1;
    if (eflag || vflag) ev_setup(eflag,vflag);
    else evflag = 0;

    int inum = list->inum;
    int nlocal = atom->nlocal;
    int newton_pair = force->newton_pair;

    vector2dc anlmtp, prod_sum_e, prod_sum_f;
    clock_t t1 = clock();
    compute_anlmtp(anlmtp);
    clock_t t2 = clock();
    compute_sum_of_prod_anlmtp(anlmtp, prod_sum_e, prod_sum_f);
    clock_t t3 = clock();

    vector2d evdwl_array(inum), fx_array(inum), fy_array(inum), fz_array(inum);
    const auto& fp = polymlp.get_fp();
    const auto& maps = polymlp.get_maps();
    const auto& type_pairs = maps.type_pairs;
    const auto& tp_to_params = maps.tp_to_params;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided)
    #endif
    for (int ii = 0; ii < inum; ii++) {
        int i,j,jnum,*jlist,type1,type2,tp,tagi,tagj;
        double delx,dely,delz,dis,evdwl,fx,fy,fz;
        dc val,valx,valy,valz,d1;
        vector1d fn,fn_d;
        vector1dc ylm,ylm_dx,ylm_dy,ylm_dz;

        double **x = atom->x;
        tagint *tag = atom->tag;

        i = list->ilist[ii];
        tagi = tag[i]-1;
        type1 = types[tagi];
        jnum = list->numneigh[i];
        jlist = list->firstneigh[i];

        evdwl_array[ii].resize(jnum);
        fx_array[ii].resize(jnum);
        fy_array[ii].resize(jnum);
        fz_array[ii].resize(jnum);

        const auto& maps_type = maps.maps_type[type1];
        const auto& nlmtp_attrs_noconj = maps_type.nlmtp_attrs_noconj;

        for (int jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            tagj = tag[j]-1;
            type2 = types[tagj];
            delx = x[i][0]-x[j][0];
            dely = x[i][1]-x[j][1];
            delz = x[i][2]-x[j][2];
            dis = sqrt(delx*delx + dely*dely + delz*delz);
            if (dis < fp.cutoff){
                tp = type_pairs[type1][type2];
                const auto& params = tp_to_params[tp];
                const vector1d diff = {delx,dely,delz};
                const vector1d &sph = cartesian_to_spherical_(diff);
                get_fn_(dis, fp, params, fn, fn_d);
                get_ylm_(dis, sph[0], sph[1], fp.maxl, ylm, ylm_dx, ylm_dy, ylm_dz);

                evdwl = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;
                const auto& ids = nlmtp_attrs_ids[type1][tp];
                const double inv_dis = 1.0 / dis;
                for (const int inlmtp: ids){
                    const auto& nlmtp = nlmtp_attrs_noconj[inlmtp];
                    const auto& lm_attr = nlmtp.lm;
                    const int nid = nlmtp.n_id;
                    const double fn_val  = fn[nid];
                    const double fn_d_val = fn_d[nid];
                    const int ylmkey = lm_attr.ylmkey;
                    const auto& ylm_val = ylm[ylmkey];
                    const auto& ylm_dx_val = ylm_dx[ylmkey];
                    const auto& ylm_dy_val = ylm_dy[ylmkey];
                    const auto& ylm_dz_val = ylm_dz[ylmkey];
                    const int idx_i = nlmtp.ilocal_noconj_id;
                    const int idx_j = nlmtp.jlocal_noconj_id;
                    val = fn_val * ylm_val;
                    d1 = fn_d_val * ylm_val * inv_dis;
                    valx = - (d1 * delx + fn_val * ylm_dx_val);
                    valy = - (d1 * dely + fn_val * ylm_dy_val);
                    valz = - (d1 * delz + fn_val * ylm_dz_val);
                    const auto& prod_ei = prod_sum_e[tagi][idx_i];
                    const auto& prod_ej = prod_sum_e[tagj][idx_j];
                    const auto& prod_fi = prod_sum_f[tagi][idx_i];
                    const auto& prod_fj = prod_sum_f[tagj][idx_j];
                    const dc sum_e = prod_ei + prod_ej * lm_attr.sign_j;
                    const dc sum_f = prod_fi + prod_fj * lm_attr.sign_j;
                    if (lm_attr.m == 0){
                        evdwl += 0.5 * prod_real(val, sum_e);
                        fx += 0.5 * prod_real(valx, sum_f);
                        fy += 0.5 * prod_real(valy, sum_f);
                        fz += 0.5 * prod_real(valz, sum_f);
                    }
                    else {
                        evdwl += prod_real(val, sum_e);
                        fx += prod_real(valx, sum_f);
                        fy += prod_real(valy, sum_f);
                        fz += prod_real(valz, sum_f);
                    }
                }
                evdwl_array[ii][jj] = evdwl;
                fx_array[ii][jj] = fx;
                fy_array[ii][jj] = fy;
                fz_array[ii][jj] = fz;
            }
        }
    }
    clock_t t4 = clock();
/*
    std::cout 
        << double(t2-t1)/CLOCKS_PER_SEC << " "
        << double(t3-t2)/CLOCKS_PER_SEC << " "
        << double(t4-t3)/CLOCKS_PER_SEC << " "
        << std::endl;
*/

    int i,j,jnum,*jlist;
    double fx,fy,fz,evdwl,dis,delx,dely,delz;
    double **f = atom->f;
    double **x = atom->x;
    for (int ii = 0; ii < inum; ii++) {
        i = list->ilist[ii];
        jnum = list->numneigh[i], jlist = list->firstneigh[i];
        for (int jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            delx = x[i][0]-x[j][0];
            dely = x[i][1]-x[j][1];
            delz = x[i][2]-x[j][2];
            dis = sqrt(delx*delx + dely*dely + delz*delz);
            if (dis < fp.cutoff){
                evdwl = evdwl_array[ii][jj];
                fx = fx_array[ii][jj]; 
                fy = fy_array[ii][jj]; 
                fz = fz_array[ii][jj]; 
                f[i][0] += fx, f[i][1] += fy, f[i][2] += fz;
                f[j][0] -= fx, f[j][1] -= fy, f[j][2] -= fz;
                if (evflag) {
                    ev_tally_xyz(i,j,nlocal,newton_pair,
                            evdwl,0.0,fx,fy,fz,delx,dely,delz);
                }
            }
        }
    }
}


void PairPolyMLP::compute_anlmtp(vector2dc& anlmtp){

    const auto& fp = polymlp.get_fp();
    const auto& maps = polymlp.get_maps();
    const auto& type_pairs = maps.type_pairs;
    const auto& tp_to_params = maps.tp_to_params;

    int inum = list->inum;
    vector2d anlmtp_r(inum), anlmtp_i(inum);
    for (int ii = 0; ii < inum; ii++){
        tagint *tag = atom->tag;
        int i = list->ilist[ii];
        int type1 = types[tag[i]-1];

        const auto& maps_type = maps.maps_type[type1];
        const auto& nlmtp_attrs_noconj = maps_type.nlmtp_attrs_noconj;
        anlmtp_r[tag[i]-1] = vector1d(nlmtp_attrs_noconj.size(), 0.0);
        anlmtp_i[tag[i]-1] = vector1d(nlmtp_attrs_noconj.size(), 0.0);
    }
    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided)
    #endif
    for (int ii = 0; ii < inum; ii++){
        int i,j,type1,type2,tp,jnum,*ilist,*jlist;
        double delx,dely,delz,dis;

        double **x = atom->x;
        tagint *tag = atom->tag;

        i = list->ilist[ii];
        type1 = types[tag[i]-1];
        jnum = list->numneigh[i];
        jlist = list->firstneigh[i];

        const auto& maps_type = maps.maps_type[type1];
        const auto& nlmtp_attrs_noconj = maps_type.nlmtp_attrs_noconj;

        vector1d fn; vector1dc ylm; dc val;
        for (int jj = 0; jj < jnum; ++jj){
            j = jlist[jj];
            delx = x[i][0]-x[j][0];
            dely = x[i][1]-x[j][1];
            delz = x[i][2]-x[j][2];
            dis = sqrt(delx*delx + dely*dely + delz*delz);
            if (dis < fp.cutoff){
                type2 = types[tag[j]-1];
                tp = type_pairs[type1][type2];
                const vector1d &sph = cartesian_to_spherical_(vector1d{delx,dely,delz});
                const auto& params = tp_to_params[tp];
                get_fn_(dis, fp, params, fn);
                get_ylm_(sph[0], sph[1], fp.maxl, ylm);

                const auto& ids = nlmtp_attrs_ids[type1][tp];
                for (int inlmtp: ids){
                    const auto& nlmtp = nlmtp_attrs_noconj[inlmtp];
                    const auto& lm_attr = nlmtp.lm;
                    const int idx_i = nlmtp.ilocal_noconj_id;
                    const int idx_j = nlmtp.jlocal_noconj_id;
                    val = fn[nlmtp.n_id] * ylm[lm_attr.ylmkey];
                    #ifdef _OPENMP
                    #pragma omp atomic
                    #endif
                    anlmtp_r[tag[i]-1][idx_i] += val.real();
                    #ifdef _OPENMP
                    #pragma omp atomic
                    #endif
                    anlmtp_r[tag[j]-1][idx_j] += val.real() * lm_attr.sign_j;
                    #ifdef _OPENMP
                    #pragma omp atomic
                    #endif
                    anlmtp_i[tag[i]-1][idx_i] += val.imag();
                    #ifdef _OPENMP
                    #pragma omp atomic
                    #endif
                    anlmtp_i[tag[j]-1][idx_j] += val.imag() * lm_attr.sign_j;
                }
            }
        }
    }
    compute_anlmtp_conjugate(anlmtp_r, anlmtp_i, anlmtp);
}

void PairPolyMLP::compute_anlmtp_conjugate(
    const vector2d& anlmtp_r, const vector2d& anlmtp_i, vector2dc& anlmtp
){

    const int inum = list->inum;
    anlmtp = vector2dc(inum);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided)
    #endif
    for (int ii = 0; ii < inum; ii++){
        tagint *tag = atom->tag;
        const int i = list->ilist[ii];
        const int type1 = types[tag[i]-1];
        polymlp.compute_anlmtp_conjugate(
            anlmtp_r[tag[i]-1], anlmtp_i[tag[i]-1], type1, anlmtp[tag[i]-1]
        );
    }
}

void PairPolyMLP::compute_sum_of_prod_anlmtp(
    const vector2dc& anlmtp, vector2dc& prod_sum_e, vector2dc& prod_sum_f
){

    const int inum = list->inum;
    prod_sum_e = vector2dc(inum);
    prod_sum_f = vector2dc(inum);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided)
    #endif
    for (int ii = 0; ii < inum; ii++) {
        tagint *tag = atom->tag;
        const int i = list->ilist[ii];
        const int type1 = types[tag[i]-1];
        polymlp.compute_sum_of_prod_anlmtp(
            anlmtp[tag[i]-1], type1, prod_sum_e[tag[i]-1], prod_sum_f[tag[i]-1]
        );
    }
}

/* ---------------------------------------------------------------------- */

void PairPolyMLP::allocate()
{

  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
      for (int j = i; j <= n; j++)
      setflag[i][j] = 0;


  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");

}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairPolyMLP::settings(int narg, char **arg)
{
// force->newton_pair = 0;
  force->newton_pair = 1;
  if (narg != 0) error->all(FLERR,"Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairPolyMLP::coeff(int narg, char **arg)
{
    if (!allocated) allocate();

    if (narg != 3 + atom->ntypes)
        error->all(FLERR,"Incorrect args for pair coefficients");

    // insure I,J args are * *
    if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
        error->all(FLERR,"Incorrect args for pair coefficients");

    std::cout << "---------- polymlp ----------" << std::endl;
    std::cout << "Parsing polymlp: " << arg[2] << std::endl;

    std::vector<std::string> ele;
    vector1d mass;
    polymlp.parse_polymlp_file(arg[2], ele, mass);

    std::cout << "Setting polymlp model required." << std::endl;

    const auto& fp = polymlp.get_fp();
    if (fp.feature_type != "gtinv" and fp.feature_type != "pair"){
        error->all(FLERR,"feature_type must be pair or gtinv");
    }       
    cutmax = fp.cutoff;
    cutforce = fp.cutoff;

    if (fp.feature_type == "gtinv") 
        set_nlmtp_attrs_ids();

    // read args that map atom types to elements in potential file
    // map[i] = which element the Ith atom type is, -1 if NULL
    map.resize(atom->ntypes);
    for (int i = 3; i < narg; i++) {
        for (int j = 0; j < ele.size(); j++){
            if (strcmp(arg[i],ele[j].c_str()) == 0){
                map[i-3] = j;
                break;
            }
        }
    }
    for (int i = 1; i <= atom->ntypes; ++i){
        atom->set_mass(FLERR,i,mass[map[i-1]]);
        for (int j = 1; j <= atom->ntypes; ++j) setflag[i][j] = 1;
    }

    std::cout << "Setting polymlp succeeded." << std::endl;
    std::cout << "-----------------------------" << std::endl;
}

void PairPolyMLP::set_types(){
    types.clear();
    for (int i = 0; i < atom->natoms; ++i){
        types.emplace_back(map[(atom->type)[i]-1]);
    }
}

void PairPolyMLP::set_nlmtp_attrs_ids(){

    const auto& fp = polymlp.get_fp();
    const auto& maps = polymlp.get_maps();
    const auto n_tp = maps.tp_to_params.size();
    for (int type1 = 0; type1 < fp.n_type; ++type1){
        const auto& maps_type = maps.maps_type[type1];
        const auto& nlmtp_attrs_noconj = maps_type.nlmtp_attrs_noconj;
        int seq = 0;
        vector2i ids(n_tp);
        for (const auto& nlmtp: nlmtp_attrs_noconj){
            ids[nlmtp.tp].emplace_back(seq);
            ++seq;
        }
        nlmtp_attrs_ids.emplace_back(ids);
    }
}


/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairPolyMLP::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");
  return cutmax;
}

/* ---------------------------------------------------------------------- */
