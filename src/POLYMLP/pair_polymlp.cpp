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
    if (pot.fp.des_type == "pair"){
        compute_pair(eflag, vflag);
    }
    else if (pot.fp.des_type == "gtinv"){
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

    vector2d antc, prod_sum_e, prod_sum_f;
    compute_antc(antc);
    compute_sum_of_prod_antc(antc, prod_sum_e, prod_sum_f);

    vector2d evdwl_array(inum), fpair_array(inum);
    for (int ii = 0; ii < inum; ii++) {
        int i = list->ilist[ii];
        int jnum = list->numneigh[i];
        evdwl_array[ii].resize(jnum);
        fpair_array[ii].resize(jnum);
    }
    const auto& ntc_map = pot.p_obj.get_ntc_map();

    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided)
    #endif
    for (int ii = 0; ii < inum; ii++) {
        int i,j,jnum,*jlist,type1,type2,tagi,tagj;
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
                get_fn_(dis, pot.fp, fn, fn_d);
                evdwl = 0.0, fpair = 0.0;
                int head_key(0);
                for (const auto& ntc: ntc_map){
                    if (type_comb[type1][type2] == ntc.tc){
                        const auto& prod_ei = prod_sum_e[tagi][head_key];
                        const auto& prod_ej = prod_sum_e[tagj][head_key];
                        const auto& prod_fi = prod_sum_f[tagi][head_key];
                        const auto& prod_fj = prod_sum_f[tagj][head_key];
                        evdwl += fn[ntc.n] * (prod_ei + prod_ej);
                        fpair += fn_d[ntc.n] * (prod_fi + prod_fj);
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

void PairPolyMLP::compute_antc(vector2d& antc){

    const auto& ntc_map = pot.p_obj.get_ntc_map();

    int inum = list->inum;
    antc = vector2d(inum, vector1d(ntc_map.size(), 0.0));

    #ifdef _OPENMP
    #pragma omp parallel for schedule(auto)
    #endif
    for (int ii = 0; ii < inum; ii++) {
        int i,j,type1,type2,jnum,*ilist,*jlist;
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
                get_fn_(dis, pot.fp, fn);
                int idx(0);
                for (const auto& ntc: ntc_map){
                    if (type_comb[type1][type2] == ntc.tc){
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        antc[tag[i]-1][idx] += fn[ntc.n];
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        antc[tag[j]-1][idx] += fn[ntc.n];
                    }
                    ++idx;
                }
            }
        }
    }
}

void PairPolyMLP::compute_sum_of_prod_antc(const vector2d& antc,
                                           vector2d& prod_antc_sum_e,
                                           vector2d& prod_antc_sum_f){

    const auto& ntc_map = pot.p_obj.get_ntc_map();

    int inum = list->inum;
    prod_antc_sum_e = vector2d(inum, vector1d(ntc_map.size(), 0.0));
    prod_antc_sum_f = vector2d(inum, vector1d(ntc_map.size(), 0.0));

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

        // computing products of order parameters (antc)
        vector1d prod_antc;
        compute_products<double>(prod_map, antc[tag[i]-1], prod_antc);

        // end: computing products of order parameters (antc)

        // computing linear features
        vector1d feature_values(linear_features.size(), 0.0);
        int idx = 0;
        for (const auto& sfeature: linear_features){
            if (sfeature.size() > 0){
                feature_values[idx] = prod_antc[sfeature[0].prod_key];
            }
            ++idx;
        }
        // end: computing linear features

        vector1d prod_features;
        compute_products<double>(prod_features_map, 
                                 feature_values, 
                                 prod_features);

        idx = 0;
        for (const auto& ntc: ntc_map){
            const auto& pmodel = pot.p_obj.get_potential_model(type1, 
                                                               ntc.ntc_key);
            double sum_e(0.0), sum_f(0.0), prod;
            for (const auto& pterm: pmodel){
                //if (fabs(prod_features[pterm.prod_features_key]) > 1e-50){
                prod = prod_antc[pterm.prod_key] 
                     * prod_features[pterm.prod_features_key];
                sum_e += pterm.coeff_e * prod;
                sum_f += pterm.coeff_f * prod;
                //}
            }
            prod_antc_sum_e[tag[i]-1][idx] = 0.5 * sum_e;
            prod_antc_sum_f[tag[i]-1][idx] = 0.5 * sum_f;
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

    vector2dc anlmtc, prod_sum_e, prod_sum_f;
    clock_t t1 = clock();
//    compute_anlmtc(anlmtc);
    compute_anlmtc_openmp(anlmtc);
    clock_t t2 = clock();
    compute_sum_of_prod_anlmtc(anlmtc, prod_sum_e, prod_sum_f);
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

    const auto& nlmtc_map_no_conj = pot.p_obj.get_nlmtc_map_no_conjugate();

    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided)
    #endif
    for (int ii = 0; ii < inum; ii++) {
        int i,j,jnum,*jlist,type1,type2,tagi,tagj;
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
                const vector1d diff = {delx,dely,delz};
                const vector1d &sph = cartesian_to_spherical_(diff);
                get_fn_(dis, pot.fp, fn, fn_d);
                get_ylm_(dis, sph[0], sph[1], pot.fp.maxl, 
                         ylm, ylm_dx, ylm_dy, ylm_dz);

                evdwl = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;
                const int tc12 = type_comb[type1][type2];
                for (const auto& nlmtc: nlmtc_map_no_conj){
                    const auto& lm_attr = nlmtc.lm;
                    const int ylmkey = lm_attr.ylmkey;
                    const int head_key = nlmtc.nlmtc_noconj_key;
                    if (tc12 == nlmtc.tc){
                        val = fn[nlmtc.n] * ylm[ylmkey];
                        d1 = fn_d[nlmtc.n] * ylm[ylmkey] / dis;
                        valx = - (d1 * delx + fn[nlmtc.n] * ylm_dx[ylmkey]);
                        valy = - (d1 * dely + fn[nlmtc.n] * ylm_dy[ylmkey]);
                        valz = - (d1 * delz + fn[nlmtc.n] * ylm_dz[ylmkey]);
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

void PairPolyMLP::compute_anlmtc(vector2dc& anlmtc){

    const auto& nlmtc_map_no_conj = pot.p_obj.get_nlmtc_map_no_conjugate();

    int inum = list->inum;
    vector2d anlmtc_r(inum, vector1d(nlmtc_map_no_conj.size(), 0.0));
    vector2d anlmtc_i(inum, vector1d(nlmtc_map_no_conj.size(), 0.0));

    for (int ii = 0; ii < inum; ii++) {
        int i,j,type1,type2,jnum,*ilist,*jlist;
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
                const vector1d &sph 
                    = cartesian_to_spherical_(vector1d{delx,dely,delz});
                get_fn_(dis, pot.fp, fn);
                get_ylm_(sph[0], sph[1], pot.fp.maxl, ylm);
                const int tc12 = type_comb[type1][type2];
                for (const auto& nlmtc: nlmtc_map_no_conj){
                    const auto& lm_attr = nlmtc.lm;
                    const int idx = nlmtc.nlmtc_noconj_key;
                    if (tc12 == nlmtc.tc){
                        val = fn[nlmtc.n] * ylm[lm_attr.ylmkey];
                        anlmtc_r[tag[i]-1][idx] += val.real();
                        anlmtc_r[tag[j]-1][idx] += val.real() * lm_attr.sign_j;
                        anlmtc_i[tag[i]-1][idx] += val.imag();
                        anlmtc_i[tag[j]-1][idx] += val.imag() * lm_attr.sign_j;
                    }
                }
            }
        }
    }
    compute_anlmtc_conjugate(anlmtc_r, anlmtc_i, anlmtc);
}

void PairPolyMLP::compute_anlmtc_openmp(vector2dc& anlmtc){

    const auto& nlmtc_map_no_conj = pot.p_obj.get_nlmtc_map_no_conjugate();

    int inum = list->inum;
    vector2d anlmtc_r(inum, vector1d(nlmtc_map_no_conj.size(), 0.0));
    vector2d anlmtc_i(inum, vector1d(nlmtc_map_no_conj.size(), 0.0));

    #ifdef _OPENMP
    #pragma omp parallel for schedule(auto)
    #endif
    for (int ii = 0; ii < inum; ii++) {
        int i,j,type1,type2,jnum,*ilist,*jlist;
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
                const vector1d &sph 
                    = cartesian_to_spherical_(vector1d{delx,dely,delz});
                get_fn_(dis, pot.fp, fn);
                get_ylm_(sph[0], sph[1], pot.fp.maxl, ylm);
                const int tc12 = type_comb[type1][type2];
                for (const auto& nlmtc: nlmtc_map_no_conj){
                    const auto& lm_attr = nlmtc.lm;
                    const int idx = nlmtc.nlmtc_noconj_key;
                    if (tc12 == nlmtc.tc){
                        val = fn[nlmtc.n] * ylm[lm_attr.ylmkey];
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        anlmtc_r[tag[i]-1][idx] += val.real();
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        anlmtc_r[tag[j]-1][idx] += val.real() * lm_attr.sign_j;
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        anlmtc_i[tag[i]-1][idx] += val.imag();
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        anlmtc_i[tag[j]-1][idx] += val.imag() * lm_attr.sign_j;
                    }
                }
            }
        }
    }
    compute_anlmtc_conjugate(anlmtc_r, anlmtc_i, anlmtc);
}

void PairPolyMLP::compute_anlmtc_conjugate(const vector2d& anlmtc_r, 
                                           const vector2d& anlmtc_i, 
                                           vector2dc& anlmtc){

    const auto& nlmtc_map_no_conj = pot.p_obj.get_nlmtc_map_no_conjugate();
    const auto& n_nlmtc_all = pot.p_obj.get_n_nlmtc_all();
    int inum = list->inum;
    anlmtc = vector2dc(inum, vector1dc(n_nlmtc_all, 0.0));

    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided)
    #endif
    for (int ii = 0; ii < anlmtc.size(); ii++) {
        tagint *tag = atom->tag;
        int i = list->ilist[ii];
        int type1 = types[tag[i]-1];
        int idx(0);
        for (const auto& nlmtc: nlmtc_map_no_conj){
            const auto& cc_coeff = nlmtc.lm.cc_coeff;
            anlmtc[ii][nlmtc.nlmtc_key] = {anlmtc_r[ii][idx], 
                                           anlmtc_i[ii][idx]};
            anlmtc[ii][nlmtc.conj_key] = {cc_coeff * anlmtc_r[ii][idx], 
                                          - cc_coeff * anlmtc_i[ii][idx]};
            ++idx;
        }
    }
}

void PairPolyMLP::compute_sum_of_prod_anlmtc(const vector2dc& anlmtc,
                                             vector2dc& prod_sum_e,
                                             vector2dc& prod_sum_f){

    const auto& nlmtc_map_no_conj = pot.p_obj.get_nlmtc_map_no_conjugate();
    const int n_head_keys = nlmtc_map_no_conj.size();
    int inum = list->inum;
    prod_sum_e = vector2dc(inum, vector1dc(n_head_keys));
    prod_sum_f = vector2dc(inum, vector1dc(n_head_keys));

    #ifdef _OPENMP
    #pragma omp parallel for schedule(auto)
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
        // computing nonequivalent products of order parameters (anlmtc)
        vector1d prod_anlmtc;
        compute_products_real(prod_map, anlmtc[tag[i]-1], prod_anlmtc);
        clock_t t2 = clock();
        // end: computing products of order parameters (anlmtc)

        // computing linear features
        //   and nonequivalent products of linear features
        vector1d features, prod_features;
        compute_linear_features(prod_anlmtc, type1, features);
        compute_products<double>(prod_features_map, features, prod_features);
        // end: computing linear features
        clock_t t3 = clock();

        vector1dc prod_anlmtc_erased;
        compute_products<dc>(prod_map_erased, 
                             anlmtc[tag[i]-1],
                             prod_anlmtc_erased);
        clock_t t4 = clock();

        for (int key = 0; key < nlmtc_map_no_conj.size(); ++key){
            const auto& pmodel = pot.p_obj.get_potential_model(type1, key);
            dc sum_e(0.0), sum_f(0.0);
//            dc prod;
            for (const auto& pterm: pmodel){
                if (fabs(prod_features[pterm.prod_features_key]) > 1e-50){
                    sum_e += pterm.coeff_e 
                           * prod_features[pterm.prod_features_key] 
                           * prod_anlmtc_erased[pterm.prod_key];
                    sum_f += pterm.coeff_f
                           * prod_features[pterm.prod_features_key] 
                           * prod_anlmtc_erased[pterm.prod_key];
                   /*
                   prod = prod_anlmtc_erased[pterm.prod_key] 
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

void PairPolyMLP::compute_linear_features(const vector1d& prod_anlmtc,
                                          const int type1,
                                          vector1d& feature_values){

    const auto& linear_features = pot.p_obj.get_linear_features(type1);
    feature_values = vector1d(linear_features.size(), 0.0);

    int idx = 0;
    double val;
    for (const auto& sfeature: linear_features){
        val = 0.0;
        for (const auto& sterm: sfeature){
            val += sterm.coeff * prod_anlmtc[sterm.prod_key];
        }
        feature_values[idx] = val;
        ++idx;
    }
}

template<typename T>
void PairPolyMLP::compute_products(const vector2i& map, 
                                   const std::vector<T>& element,
                                   std::vector<T>& prod_vals){

    prod_vals = std::vector<T>(map.size());

    int idx(0);
    T val_p;
    for (const auto& prod: map){
        if (prod.size() > 0){
            auto iter = prod.begin();
            val_p = element[*iter];
            ++iter;
            while (iter != prod.end()){
                val_p *= element[*iter];
                ++iter;
            }
        }
        else val_p = 1.0;

        prod_vals[idx] = val_p;
        ++idx;
    }
}

void PairPolyMLP::compute_products_real(const vector2i& map, 
                                        const vector1dc& element,
                                        vector1d& prod_vals){

    prod_vals = vector1d(map.size());

    int idx(0);
    dc val_p; 
    for (const auto& prod: map){
        if (prod.size() > 1) {
            auto iter = prod.begin() + 1;
            val_p = element[*iter];
            ++iter;
            while (iter != prod.end()){
                val_p *= element[*iter];
                ++iter;
            }
            prod_vals[idx] = prod_real(val_p, element[*(prod.begin())]);
        }
        else if (prod.size() == 1){
            prod_vals[idx] = element[*(prod.begin())].real();
        }
        else prod_vals[idx] = 1.0;
        ++idx;
    }
}


double PairPolyMLP::prod_real(const dc& val1, const dc& val2){
    return val1.real() * val2.real() - val1.imag() * val2.imag();
}

dc PairPolyMLP::prod_real_and_complex(const double val1, const dc& val2){
    return dc(val1 * val2.real(), val1 * val2.imag());
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
    std::ifstream input(file);
    if (input.fail()){
        std::cerr << "Error: Could not open mlp file: " << file << "\n";
        exit(8);
    }

    std::stringstream ss;
    std::string line, tmp;

    // line 1: elements
    ele.clear();
    std::getline( input, line );
    ss << line;
    while (!ss.eof()){
        ss >> tmp;
        ele.push_back(tmp);
    }
    ele.erase(ele.end()-1);
    ele.erase(ele.end()-1);
    ss.str("");
    ss.clear(std::stringstream::goodbit);

    pot.fp.n_type = int(ele.size());
    pot.fp.force = true;

    // line 2-4: cutoff radius, pair type, descriptor type
    // line 5-7: model_type, max power, max l
    pot.fp.cutoff = cutmax = cutforce = get_value<double>(input);
    pot.fp.pair_type = get_value<std::string>(input);
    pot.fp.des_type = get_value<std::string>(input);
    pot.fp.model_type = get_value<int>(input);
    pot.fp.maxp = get_value<int>(input);
    pot.fp.maxl = get_value<int>(input);

    if (pot.fp.des_type != "gtinv" and pot.fp.des_type != "pair"){
        error->all(FLERR,"des_type must be pair or gtinv");
    }

    // line 8-10: gtinv_order, gtinv_maxl and gtinv_sym (optional)
    if (pot.fp.des_type == "gtinv"){
        int gtinv_order = get_value<int>(input);
        int size = gtinv_order - 1;
        vector1i gtinv_maxl = get_value_array<int>(input, size);
        std::vector<bool> gtinv_sym = get_value_array<bool>(input, size);

        Readgtinv rgt(gtinv_order, gtinv_maxl, gtinv_sym, ele.size());
        pot.fp.lm_array = rgt.get_lm_seq();
        pot.fp.l_comb = rgt.get_l_comb();
        pot.fp.lm_coeffs = rgt.get_lm_coeffs(); 
    }

    // line 11: number of regression coefficients
    // line 12,13: regression coefficients, scale coefficients
    int n_reg_coeffs = get_value<int>(input);
    vector1d reg_coeffs = get_value_array<double>(input, n_reg_coeffs);
    vector1d scale = get_value_array<double>(input, n_reg_coeffs);
    for (int i = 0; i < n_reg_coeffs; ++i) reg_coeffs[i] *= 2.0/scale[i];

    // line 14: number of gaussian parameters
    // line 15-: gaussian parameters
    int n_params = get_value<int>(input);
    pot.fp.params = vector2d(n_params);
    for (int i = 0; i < n_params; ++i)
        pot.fp.params[i] = get_value_array<double>(input, 2);
    
    // last line: atomic mass
    mass = get_value_array<double>(input, ele.size());

    const bool icharge = false;
    pot.modelp = ModelParams(pot.fp, icharge);
    const Features f_obj(pot.fp, pot.modelp);
    pot.p_obj = Potential(f_obj, reg_coeffs);

    type_comb = vector2i(pot.fp.n_type, vector1i(pot.fp.n_type));
    for (int type1 = 0; type1 < pot.fp.n_type; ++type1){
        for (int type2 = 0; type2 < pot.fp.n_type; ++type2){
            for (int i = 0; i < pot.modelp.get_type_comb_pair().size(); ++i){
                const auto &tc = pot.modelp.get_type_comb_pair()[i];
                if (tc[type1].size() > 0 and tc[type1][0] == type2){
                    type_comb[type1][type2] = i;
                    break;
                }
            }
        }
    }
}

template<typename T>
T PairPolyMLP::get_value(std::ifstream& input){

    std::string line;
    std::stringstream ss;

    T val;
    std::getline( input, line );
    ss << line;
    ss >> val;

    return val;
}

template<typename T>
std::vector<T> PairPolyMLP::get_value_array(std::ifstream& input, 
                                            const int& size){

    std::string line;
    std::stringstream ss;

    std::vector<T> array(size);

    std::getline( input, line );
    ss << line;
    T val;
    for (int i = 0; i < array.size(); ++i){
        ss >> val;
        array[i] = val;
    }

    return array;
}

