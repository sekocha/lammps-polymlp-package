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
    if (pot.fp.feature_type == "pair"){
        compute_pair(eflag, vflag);
    }
    else if (pot.fp.feature_type == "gtinv"){
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
    for (int ii = 0; ii < inum; ii++) {
        int i = list->ilist[ii];
        int jnum = list->numneigh[i];
        evdwl_array[ii].resize(jnum);
        fpair_array[ii].resize(jnum);
    }
    const auto& ntp_attrs = pot.mapping.get_ntp_attrs();
    const auto& tp_to_params = pot.mapping.get_type_pair_to_params();

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

        vector1d fn,fn_d;
        for (int jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            tagj = tag[j]-1;
            type2 = types[tagj];
            delx = x[i][0]-x[j][0];
            dely = x[i][1]-x[j][1];
            delz = x[i][2]-x[j][2];
            dis = sqrt(delx*delx + dely*dely + delz*delz);
            if (dis < pot.fp.cutoff){
                tp = type_pairs[type1][type2];
                const auto& params = tp_to_params[tp];
                get_fn_(dis, pot.fp, params, fn, fn_d);
                evdwl = 0.0, fpair = 0.0;
                int head_key(0);
                for (const auto& ntp: ntp_attrs){
                    if (tp == ntp.tp){
                        const auto& prod_ei = prod_sum_e[tagi][head_key];
                        const auto& prod_ej = prod_sum_e[tagj][head_key];
                        const auto& prod_fi = prod_sum_f[tagi][head_key];
                        const auto& prod_fj = prod_sum_f[tagj][head_key];
                        evdwl += fn[ntp.n_id] * (prod_ei + prod_ej);
                        fpair += fn_d[ntp.n_id] * (prod_fi + prod_fj);
                    }
                    ++head_key;
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
            if (dis < pot.fp.cutoff){
                evdwl = evdwl_array[ii][jj];
                fpair = fpair_array[ii][jj];
                f[i][0] += fpair*delx;
                f[i][1] += fpair*dely;
                f[i][2] += fpair*delz;
                //            if (newton_pair || j < nlocal)
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

    const auto& ntp_attrs = pot.mapping.get_ntp_attrs();
    const auto& tp_to_params = pot.mapping.get_type_pair_to_params();

    int inum = list->inum;
    antp = vector2d(inum, vector1d(ntp_attrs.size(), 0.0));

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

        vector1d fn; 
        for (int jj = 0; jj < jnum; ++jj) {
            j = jlist[jj];
            delx = x[i][0]-x[j][0];
            dely = x[i][1]-x[j][1];
            delz = x[i][2]-x[j][2];
            dis = sqrt(delx*delx + dely*dely + delz*delz);
            if (dis < pot.fp.cutoff){
                type2 = types[tag[j]-1];
                tp = type_pairs[type1][type2];
                const auto& params = tp_to_params[tp];
                get_fn_(dis, pot.fp, params, fn);
                int idx(0);
                for (const auto& ntp: ntp_attrs){
                    if (tp == ntp.tp){
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        antp[tag[i]-1][idx] += fn[ntp.n_id];
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        antp[tag[j]-1][idx] += fn[ntp.n_id];
                    }
                    ++idx;
                }
            }
        }
    }
}

void PairPolyMLP::compute_sum_of_prod_antp(
    const vector2d& antp, vector2d& prod_antp_sum_e, vector2d& prod_antp_sum_f
){

    const auto& ntp_attrs = pot.mapping.get_ntp_attrs();

    int inum = list->inum;
    prod_antp_sum_e = vector2d(inum, vector1d(ntp_attrs.size(), 0.0));
    prod_antp_sum_f = vector2d(inum, vector1d(ntp_attrs.size(), 0.0));

    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided)
    #endif
    for (int ii = 0; ii < inum; ii++) {
        int i,type1,*ilist;
        tagint *tag = atom->tag;
        i = list->ilist[ii];
        type1 = types[tag[i]-1];

        const auto& linear_features = pot.p_obj.get_linear_features(type1);
        const auto& prod_map = pot.p_obj.get_prod_map(type1);
        const auto& prod_features_map = pot.p_obj.get_prod_features_map(type1);

        // computing products of order parameters (antp)
        vector1d prod_antp;
        compute_products<double>(prod_map, antp[tag[i]-1], prod_antp);

        // end: computing products of order parameters (antp)

        // computing linear features
        vector1d feature_values(linear_features.size(), 0.0);
        int idx = 0;
        for (const auto& sfeature: linear_features){
            if (sfeature.size() > 0){
                feature_values[idx] = prod_antp[sfeature[0].prod_key];
            }
            ++idx;
        }
        // end: computing linear features

        vector1d prod_features;
        compute_products<double>(prod_features_map, 
                                 feature_values, 
                                 prod_features);

        idx = 0;
        for (const auto& ntp: ntp_attrs){
            const auto& pmodel = pot.p_obj.get_potential_model(type1, ntp.ntp_key);
            double sum_e(0.0), sum_f(0.0), prod;
            for (const auto& pterm: pmodel){
                //if (fabs(prod_features[pterm.prod_features_key]) > 1e-50){
                prod = prod_antp[pterm.prod_key] 
                     * prod_features[pterm.prod_features_key];
                sum_e += pterm.coeff_e * prod;
                sum_f += pterm.coeff_f * prod;
                //}
            }
            prod_antp_sum_e[tag[i]-1][idx] = 0.5 * sum_e;
            prod_antp_sum_f[tag[i]-1][idx] = 0.5 * sum_f;
            ++idx;
        }
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
//    compute_anlmtp(anlmtp);
    compute_anlmtp_openmp(anlmtp);
    clock_t t2 = clock();
    compute_sum_of_prod_anlmtp(anlmtp, prod_sum_e, prod_sum_f);
    clock_t t3 = clock();

    vector2d evdwl_array(inum),fx_array(inum),fy_array(inum),fz_array(inum);
    for (int ii = 0; ii < inum; ii++) {
        int i = list->ilist[ii];
        int jnum = list->numneigh[i];
        evdwl_array[ii].resize(jnum);
        fx_array[ii].resize(jnum);
        fy_array[ii].resize(jnum);
        fz_array[ii].resize(jnum);
    }

    const auto& nlmtp_attrs_no_conj = pot.mapping.get_nlmtp_attrs_no_conjugate();
    const auto& tp_to_params = pot.mapping.get_type_pair_to_params();

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

        for (int jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            tagj = tag[j]-1;
            type2 = types[tagj];
            delx = x[i][0]-x[j][0];
            dely = x[i][1]-x[j][1];
            delz = x[i][2]-x[j][2];
            dis = sqrt(delx*delx + dely*dely + delz*delz);
            if (dis < pot.fp.cutoff){
                tp = type_pairs[type1][type2];
                const auto& params = tp_to_params[tp];
                const vector1d diff = {delx,dely,delz};
                const vector1d &sph = cartesian_to_spherical_(diff);
                get_fn_(dis, pot.fp, params, fn, fn_d);
                get_ylm_(dis, sph[0], sph[1], pot.fp.maxl, 
                         ylm, ylm_dx, ylm_dy, ylm_dz);

                evdwl = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;
                const int tp = type_pairs[type1][type2];
                for (const auto& nlmtp: nlmtp_attrs_no_conj){
                    const auto& lm_attr = nlmtp.lm;
                    const int ylmkey = lm_attr.ylmkey;
                    const int head_key = nlmtp.nlmtp_noconj_key;
                    if (tp == nlmtp.tp){
                        val = fn[nlmtp.n_id] * ylm[ylmkey];
                        d1 = fn_d[nlmtp.n_id] * ylm[ylmkey] / dis;
                        valx = - (d1 * delx + fn[nlmtp.n_id] * ylm_dx[ylmkey]);
                        valy = - (d1 * dely + fn[nlmtp.n_id] * ylm_dy[ylmkey]);
                        valz = - (d1 * delz + fn[nlmtp.n_id] * ylm_dz[ylmkey]);
                        const auto& prod_ei = prod_sum_e[tagi][head_key];
                        const auto& prod_ej = prod_sum_e[tagj][head_key];
                        const auto& prod_fi = prod_sum_f[tagi][head_key];
                        const auto& prod_fj = prod_sum_f[tagj][head_key];
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
            if (dis < pot.fp.cutoff){
                evdwl = evdwl_array[ii][jj];
                fx = fx_array[ii][jj]; 
                fy = fy_array[ii][jj]; 
                fz = fz_array[ii][jj]; 
                f[i][0] += fx, f[i][1] += fy, f[i][2] += fz;
                // if (newton_pair || j < nlocal)
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

    const auto& nlmtp_attrs_no_conj = pot.mapping.get_nlmtp_attrs_no_conjugate();
    const auto& tp_to_params = pot.mapping.get_type_pair_to_params();

    int inum = list->inum;
    vector2d anlmtp_r(inum, vector1d(nlmtp_attrs_no_conj.size(), 0.0));
    vector2d anlmtp_i(inum, vector1d(nlmtp_attrs_no_conj.size(), 0.0));

    for (int ii = 0; ii < inum; ii++) {
        int i,j,type1,type2,tp,jnum,*ilist,*jlist;
        double delx,dely,delz,dis;

        double **x = atom->x;
        tagint *tag = atom->tag;

        i = list->ilist[ii];
        type1 = types[tag[i]-1];
        jnum = list->numneigh[i];
        jlist = list->firstneigh[i];

        vector1d fn; vector1dc ylm; dc val;
        for (int jj = 0; jj < jnum; ++jj) {
            j = jlist[jj];
            delx = x[i][0]-x[j][0];
            dely = x[i][1]-x[j][1];
            delz = x[i][2]-x[j][2];
            dis = sqrt(delx*delx + dely*dely + delz*delz);
            if (dis < pot.fp.cutoff){
                type2 = types[tag[j]-1];
                tp = type_pairs[type1][type2];
                const vector1d &sph 
                    = cartesian_to_spherical_(vector1d{delx,dely,delz});
                const auto& params = tp_to_params[tp];
                get_fn_(dis, pot.fp, params, fn);
                get_ylm_(sph[0], sph[1], pot.fp.maxl, ylm);
                const int tc12 = type_pairs[type1][type2];
                for (const auto& nlmtp: nlmtp_attrs_no_conj){
                    const auto& lm_attr = nlmtp.lm;
                    const int idx = nlmtp.nlmtp_noconj_key;
                    if (tp == nlmtp.tp){
                        val = fn[nlmtp.n_id] * ylm[lm_attr.ylmkey];
                        anlmtp_r[tag[i]-1][idx] += val.real();
                        anlmtp_r[tag[j]-1][idx] += val.real() * lm_attr.sign_j;
                        anlmtp_i[tag[i]-1][idx] += val.imag();
                        anlmtp_i[tag[j]-1][idx] += val.imag() * lm_attr.sign_j;
                    }
                }
            }
        }
    }
    compute_anlmtp_conjugate(anlmtp_r, anlmtp_i, anlmtp);
}

void PairPolyMLP::compute_anlmtp_openmp(vector2dc& anlmtp){

    const auto& nlmtp_attrs_no_conj = pot.mapping.get_nlmtp_attrs_no_conjugate();
    const auto& tp_to_params = pot.mapping.get_type_pair_to_params();

    int inum = list->inum;
    vector2d anlmtp_r(inum, vector1d(nlmtp_attrs_no_conj.size(), 0.0));
    vector2d anlmtp_i(inum, vector1d(nlmtp_attrs_no_conj.size(), 0.0));

    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided)
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

        vector1d fn; vector1dc ylm; dc val;
        for (int jj = 0; jj < jnum; ++jj) {
            j = jlist[jj];
            delx = x[i][0]-x[j][0];
            dely = x[i][1]-x[j][1];
            delz = x[i][2]-x[j][2];
            dis = sqrt(delx*delx + dely*dely + delz*delz);
            if (dis < pot.fp.cutoff){
                type2 = types[tag[j]-1];
                tp = type_pairs[type1][type2];
                const vector1d &sph 
                    = cartesian_to_spherical_(vector1d{delx,dely,delz});
                const auto& params = tp_to_params[tp];
                get_fn_(dis, pot.fp, params, fn);
                get_ylm_(sph[0], sph[1], pot.fp.maxl, ylm);
                const int tc12 = type_pairs[type1][type2];
                for (const auto& nlmtp: nlmtp_attrs_no_conj){
                    const auto& lm_attr = nlmtp.lm;
                    const int idx = nlmtp.nlmtp_noconj_key;
                    if (tp == nlmtp.tp){
                        val = fn[nlmtp.n_id] * ylm[lm_attr.ylmkey];
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        anlmtp_r[tag[i]-1][idx] += val.real();
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        anlmtp_r[tag[j]-1][idx] += val.real() * lm_attr.sign_j;
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        anlmtp_i[tag[i]-1][idx] += val.imag();
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        anlmtp_i[tag[j]-1][idx] += val.imag() * lm_attr.sign_j;
                    }
                }
            }
        }
    }
    compute_anlmtp_conjugate(anlmtp_r, anlmtp_i, anlmtp);
}

void PairPolyMLP::compute_anlmtp_conjugate(const vector2d& anlmtp_r, 
                                           const vector2d& anlmtp_i, 
                                           vector2dc& anlmtp){

    const auto& nlmtp_attrs_no_conj = pot.mapping.get_nlmtp_attrs_no_conjugate();
    const auto& n_nlmtp_all = pot.mapping.get_n_nlmtp_all();
    int inum = list->inum;
    anlmtp = vector2dc(inum, vector1dc(n_nlmtp_all, 0.0));

    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided)
    #endif
    for (int ii = 0; ii < anlmtp.size(); ii++) {
        tagint *tag = atom->tag;
        int i = list->ilist[ii];
        int type1 = types[tag[i]-1];
        int idx(0);
        for (const auto& nlmtp: nlmtp_attrs_no_conj){
            const auto& cc_coeff = nlmtp.lm.cc_coeff;
            anlmtp[ii][nlmtp.nlmtp_key] = {anlmtp_r[ii][idx], anlmtp_i[ii][idx]};
            anlmtp[ii][nlmtp.conj_key] = {cc_coeff * anlmtp_r[ii][idx], 
                                          - cc_coeff * anlmtp_i[ii][idx]};
            ++idx;
        }
    }
}

void PairPolyMLP::compute_sum_of_prod_anlmtp(const vector2dc& anlmtp,
                                             vector2dc& prod_sum_e,
                                             vector2dc& prod_sum_f){

    const auto& nlmtp_attrs_no_conj = pot.mapping.get_nlmtp_attrs_no_conjugate();
    const int n_head_keys = nlmtp_attrs_no_conj.size();
    int inum = list->inum;
    prod_sum_e = vector2dc(inum, vector1dc(n_head_keys));
    prod_sum_f = vector2dc(inum, vector1dc(n_head_keys));

    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided)
    #endif
    for (int ii = 0; ii < inum; ii++) {
        int i,type1,*ilist;
        tagint *tag = atom->tag;
        i = list->ilist[ii];
        type1 = types[tag[i]-1];

        const auto& prod_map = pot.p_obj.get_prod_map(type1);
        const auto& prod_map_erased = pot.p_obj.get_prod_map_erased(type1);
        const auto& prod_features_map = pot.p_obj.get_prod_features_map(type1);

        clock_t t1 = clock();
        // computing nonequivalent products of order parameters (anlmtp)
        vector1d prod_anlmtp;
        compute_products_real(prod_map, anlmtp[tag[i]-1], prod_anlmtp);
        clock_t t2 = clock();
        // end: computing products of order parameters (anlmtp)

        // computing linear features
        //   and nonequivalent products of linear features
        vector1d features, prod_features;
        compute_linear_features(prod_anlmtp, type1, features);
        compute_products<double>(prod_features_map, features, prod_features);
        // end: computing linear features
        clock_t t3 = clock();

        vector1dc prod_anlmtp_erased;
        compute_products<dc>(prod_map_erased, 
                             anlmtp[tag[i]-1],
                             prod_anlmtp_erased);
        clock_t t4 = clock();

        for (int key = 0; key < nlmtp_attrs_no_conj.size(); ++key){
            const auto& pmodel = pot.p_obj.get_potential_model(type1, key);
            dc sum_e(0.0), sum_f(0.0);
//            dc prod;
            for (const auto& pterm: pmodel){
                // TODO: examine accuracy
                if (fabs(prod_features[pterm.prod_features_key]) > 1e-50){
                    sum_e += pterm.coeff_e 
                           * prod_features[pterm.prod_features_key] 
                           * prod_anlmtp_erased[pterm.prod_key];
                    sum_f += pterm.coeff_f
                           * prod_features[pterm.prod_features_key] 
                           * prod_anlmtp_erased[pterm.prod_key];
                   /*
                   prod = prod_anlmtp_erased[pterm.prod_key] 
                         * prod_features[pterm.prod_features_key];
                   sum_e += pterm.coeff_e * prod;
                   sum_f += pterm.coeff_f * prod;
                   */
                }
            }
            prod_sum_e[tag[i]-1][key] = sum_e;
            prod_sum_f[tag[i]-1][key] = sum_f;
        }

        clock_t t5 = clock();
    /* 
        std::cout 
            << double(t2-t1)/CLOCKS_PER_SEC << " "
            << double(t3-t2)/CLOCKS_PER_SEC << " "
            << double(t4-t3)/CLOCKS_PER_SEC << " "
            << double(t5-t4)/CLOCKS_PER_SEC << " "
            << std::endl;
            */
    }
}

void PairPolyMLP::compute_linear_features(const vector1d& prod_anlmtp,
                                          const int type1,
                                          vector1d& feature_values){

    const auto& linear_features = pot.p_obj.get_linear_features(type1);
    feature_values = vector1d(linear_features.size(), 0.0);

    int idx = 0;
    double val;
    for (const auto& sfeature: linear_features){
        val = 0.0;
        for (const auto& sterm: sfeature){
            val += sterm.coeff * prod_anlmtp[sterm.prod_key];
        }
        feature_values[idx] = val;
        ++idx;
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

    read_pot(arg[2]);

    // read args that map atom types to elements in potential file
    // map[i] = which element the Ith atom type is, -1 if NULL
    std::vector<int> map(atom->ntypes);
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

    for (int i = 0; i < atom->natoms; ++i){
        types.emplace_back(map[(atom->type)[i]-1]);
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

void PairPolyMLP::read_pot(char *file)
{

    const bool legacy = check_polymlp_legacy(file);
    vector1d reg_coeffs;
    if (legacy == true){
        parse_polymlp_legacy(file, pot.fp, reg_coeffs, ele, mass);
    }
    else {
        parse_polymlp(file, pot.fp, reg_coeffs, ele, mass);
    }

    if (pot.fp.feature_type != "gtinv" and pot.fp.feature_type != "pair"){
        error->all(FLERR,"feature_type must be pair or gtinv");
    }       
    cutmax = pot.fp.cutoff;
    cutforce = pot.fp.cutoff;

    const Features f_obj(pot.fp);
    pot.mapping = f_obj.get_mapping();
    pot.modelp = f_obj.get_model_params();
    pot.p_obj = Potential(f_obj, reg_coeffs);
    type_pairs = pot.mapping.get_type_pairs();

}

