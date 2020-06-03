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

#include "pair_mlip_gtinv.h"

using namespace LAMMPS_NS;

#define MAXLINE 1024
#define DELTA 4

/* ---------------------------------------------------------------------- */

PairMLIPGtinv::PairMLIPGtinv(LAMMPS *lmp) : Pair(lmp)
{
    restartinfo = 0;
}


/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairMLIPGtinv::~PairMLIPGtinv()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

/* ---------------------------------------------------------------------- */

void PairMLIPGtinv::compute(int eflag, int vflag)
{

    vflag = 1;
    if (eflag || vflag) ev_setup(eflag,vflag);
    else evflag = 0;

    int inum = list->inum;
    int nlocal = atom->nlocal;
    int newton_pair = force->newton_pair;

    const int n_type_comb = pot.modelp.get_type_comb_pair().size();
    const int n_fn = pot.modelp.get_n_fn(); 
    const int n_lm = pot.lm_info.size();
    const int n_lm_all = 2 * n_lm - pot.fp.maxl - 1;

    barray4dc prod_anlm_f(boost::extents[n_type_comb][inum][n_fn][n_lm_all]);
    barray4dc prod_anlm_e(boost::extents[n_type_comb][inum][n_fn][n_lm_all]);

    const barray4dc &anlm = compute_anlm();
    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided)
    #endif
    for (int ii = 0; ii < inum; ii++) {
        int i,type1;
        double regc,valreal,valimag;
        dc valtmp;

        tagint *tag = atom->tag;
        i = list->ilist[ii], type1 = types[tag[i]-1];

        const int n_gtinv = pot.modelp.get_linear_term_gtinv().size();
        const vector2dc &uniq 
            = compute_anlm_uniq_products(type1, anlm[tag[i]-1]);
        vector1d uniq_p;
        if (pot.fp.maxp > 1){
            uniq_p = compute_polynomial_model_uniq_products
                (type1, anlm[tag[i]-1], uniq);
        }

        for (int type2 = 0; type2 < pot.fp.n_type; ++type2){
            const int tc0 = type_comb[type1][type2];
            for (int n = 0; n < n_fn; ++n){
                for (int lm0 = 0; lm0 < n_lm_all; ++lm0){
                    dc sumf(0.0), sume(0.0);
                    for (auto& inv: pot.poly_gtinv.get_gtinv_info(tc0,lm0)){
                        regc = 0.5 * pot.reg_coeffs[n * n_gtinv + inv.reg_i];
                        if (inv.lmt_pi != -1){
                            valtmp = regc * inv.coeff * uniq[n][inv.lmt_pi];
                            valreal = valtmp.real() / inv.order;
                            valimag = valtmp.imag() / inv.order;
                            sumf += valtmp;
                            sume += dc({valreal,valimag});
                        }
                        else {
                            sumf += regc;
                            sume += regc;
                        }
                    }
                    // polynomial model correction
                    if (pot.fp.maxp > 1){
                        for (const auto& pi: 
                            pot.poly_gtinv.get_polynomial_info(tc0,n,lm0)){
                            regc = pot.reg_coeffs[pi.reg_i] * uniq_p[pi.comb_i];
                            if (pi.lmt_pi != -1){
                                valtmp = regc * pi.coeff * uniq[n][pi.lmt_pi];
                                valreal = valtmp.real() / pi.order;
                                valimag = valtmp.imag() / pi.order;
                                sumf += valtmp;
                                sume += dc({valreal,valimag});
                            }
                            else {
                                sumf += regc;
                                sume += regc / pi.order;
                            }
                        }
                    }
                    // end: polynomial model correction
                    prod_anlm_f[tc0][tag[i]-1][n][lm0] = sumf;
                    prod_anlm_e[tc0][tag[i]-1][n][lm0] = sume;
                }
            }
        }
    }

    vector2d evdwl_array(inum),fx_array(inum),fy_array(inum),fz_array(inum);
    for (int ii = 0; ii < inum; ii++) {
        int i = list->ilist[ii];
        int jnum = list->numneigh[i];
        evdwl_array[ii].resize(jnum);
        fx_array[ii].resize(jnum);
        fy_array[ii].resize(jnum);
        fz_array[ii].resize(jnum);
    }

    vector1d scales;
    for (int l = 0; l < pot.fp.maxl+1; ++l){
        if (l%2 == 0) for (int m = -l; m < l+1; ++m) scales.emplace_back(1.0);
        else for (int m = -l; m < l+1; ++m) scales.emplace_back(-1.0);
    }

    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided)
    #endif
    for (int ii = 0; ii < inum; ii++) {
        int i,j,jnum,*jlist,type1,type2,tc,m,lm1,lm2;
        double delx,dely,delz,dis,evdwl,fx,fy,fz,
               costheta,sintheta,cosphi,sinphi,coeff,cc;
        dc f1,ylm_dphi,d0,d1,d2,term1,term2,sume,sumf;

        double **x = atom->x;
        tagint *tag = atom->tag;

        i = list->ilist[ii];
        type1 = types[tag[i]-1];
        jnum = list->numneigh[i];
        jlist = list->firstneigh[i];

        const int n_fn = pot.modelp.get_n_fn();
        const int n_des = pot.modelp.get_n_des();
        const int n_lm = pot.lm_info.size();
        const int n_lm_all = 2 * n_lm - pot.fp.maxl - 1;
        const int n_gtinv = pot.modelp.get_linear_term_gtinv().size();

        vector1d fn,fn_d;
        vector1dc ylm,ylm_dtheta;
        vector2dc fn_ylm,fn_ylm_dx,fn_ylm_dy,fn_ylm_dz;

        fn_ylm = fn_ylm_dx = fn_ylm_dy = fn_ylm_dz 
            = vector2dc(n_fn, vector1dc(n_lm_all));

        for (int jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            delx = x[i][0]-x[j][0];
            dely = x[i][1]-x[j][1];
            delz = x[i][2]-x[j][2];
            dis = sqrt(delx*delx + dely*dely + delz*delz);

            if (dis < pot.fp.cutoff){
                type2 = types[tag[j]-1];

                const vector1d &sph 
                    = cartesian_to_spherical(vector1d{delx,dely,delz});
                get_fn(dis, pot.fp, fn, fn_d);
                get_ylm(sph, pot.lm_info, ylm, ylm_dtheta);

                costheta = cos(sph[0]), sintheta = sin(sph[0]);
                cosphi = cos(sph[1]), sinphi = sin(sph[1]);
                fabs(sintheta) > 1e-15 ? 
                    (coeff = 1.0 / sintheta) : (coeff = 0);
                for (int lm = 0; lm < n_lm; ++lm) {
                    m = pot.lm_info[lm][1], lm1 = pot.lm_info[lm][2], 
                      lm2 = pot.lm_info[lm][3];
                    cc = pow(-1, m); 
                    ylm_dphi = dc{0.0,1.0} * double(m) * ylm[lm];
                    term1 = ylm_dtheta[lm] * costheta;
                    term2 = coeff * ylm_dphi;
                    d0 = term1 * cosphi - term2 * sinphi;
                    d1 = term1 * sinphi + term2 * cosphi;
                    d2 = - ylm_dtheta[lm] * sintheta;
                    for (int n = 0; n < n_fn; ++n) {
                        fn_ylm[n][lm1] = fn[n] * ylm[lm];
                        fn_ylm[n][lm2] = cc * std::conj(fn_ylm[n][lm1]);
                        f1 = fn_d[n] * ylm[lm];
                        fn_ylm_dx[n][lm1] = - (f1 * delx + fn[n] * d0) / dis;
                        fn_ylm_dx[n][lm2] = cc * std::conj(fn_ylm_dx[n][lm1]);
                        fn_ylm_dy[n][lm1] = - (f1 * dely + fn[n] * d1) / dis;
                        fn_ylm_dy[n][lm2] = cc * std::conj(fn_ylm_dy[n][lm1]);
                        fn_ylm_dz[n][lm1] = - (f1 * delz + fn[n] * d2) / dis;
                        fn_ylm_dz[n][lm2] = cc * std::conj(fn_ylm_dz[n][lm1]);
                    }
                }

                const int tc0 = type_comb[type1][type2];
                const auto &prodif = prod_anlm_f[tc0][tag[i]-1];
                const auto &prodie = prod_anlm_e[tc0][tag[i]-1];
                const auto &prodjf = prod_anlm_f[tc0][tag[j]-1];
                const auto &prodje = prod_anlm_e[tc0][tag[j]-1];

                evdwl = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;
                // including polynomial correction
                for (int n = 0; n < n_fn; ++n) {
                    for (int lm0 = 0; lm0 < n_lm_all; ++lm0) {
                        sume = prodie[n][lm0] + prodje[n][lm0] * scales[lm0];
                        sumf = prodif[n][lm0] + prodjf[n][lm0] * scales[lm0];
                        evdwl += prod_real(fn_ylm[n][lm0], sume);
                        fx += prod_real(fn_ylm_dx[n][lm0], sumf);
                        fy += prod_real(fn_ylm_dy[n][lm0], sumf);
                        fz += prod_real(fn_ylm_dz[n][lm0], sumf);
                   }
                }
                evdwl_array[ii][jj] = evdwl;
                fx_array[ii][jj] = fx;
                fy_array[ii][jj] = fy;
                fz_array[ii][jj] = fz;
            }
        }
    }

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


barray4dc PairMLIPGtinv::compute_anlm(){

    const int n_fn = pot.modelp.get_n_fn(), n_lm = pot.lm_info.size(), 
        n_lm_all = 2 * n_lm - pot.fp.maxl - 1, n_type = pot.fp.n_type;

    int inum = list->inum;
    int nlocal = atom->nlocal;
    int newton_pair = force->newton_pair;

    barray4dc anlm(boost::extents[inum][n_type][n_fn][n_lm_all]);
    barray4d anlm_r(boost::extents[inum][n_type][n_fn][n_lm]);
    barray4d anlm_i(boost::extents[inum][n_type][n_fn][n_lm]);
    std::fill(anlm_r.data(), anlm_r.data() + anlm_r.num_elements(), 0.0);
    std::fill(anlm_i.data(), anlm_i.data() + anlm_i.num_elements(), 0.0);
    
    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided)
    #endif
    for (int ii = 0; ii < inum; ii++) {
        int i,j,type1,type2,jnum,sindex,*ilist,*jlist;
        double delx,dely,delz,dis,scale;

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
                    = cartesian_to_spherical(vector1d{delx,dely,delz});
                get_fn(dis, pot.fp, fn);
                get_ylm(sph, pot.lm_info, ylm);
                for (int n = 0; n < n_fn; ++n) {
                    for (int lm = 0; lm < n_lm; ++lm) {
                        if (pot.lm_info[lm][0]%2 == 0) scale = 1.0;
                        else scale = -1.0;
                        val = fn[n] * ylm[lm];
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        anlm_r[tag[i]-1][type2][n][lm] += val.real();
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        anlm_r[tag[j]-1][type1][n][lm] += val.real() * scale;
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        anlm_i[tag[i]-1][type2][n][lm] += val.imag();
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        anlm_i[tag[j]-1][type1][n][lm] += val.imag() * scale;
                    }
                }
            }
        }
    }

    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided)
    #endif
    for (int ii = 0; ii < inum; ii++) {
        int m,lm1,lm2;
        double cc;
        for (int type2 = 0; type2 < n_type; ++type2) {
            for (int n = 0; n < n_fn; ++n) {
                for (int lm = 0; lm < n_lm; ++lm) {
                    m = pot.lm_info[lm][1];
                    lm1 = pot.lm_info[lm][2], lm2 = pot.lm_info[lm][3];
                    anlm[ii][type2][n][lm1] = 
                        {anlm_r[ii][type2][n][lm], anlm_i[ii][type2][n][lm]};
                    cc = pow(-1, m); 
                    anlm[ii][type2][n][lm2] =
                        {cc * anlm_r[ii][type2][n][lm],
                        - cc * anlm_i[ii][type2][n][lm]};
                }
            }
        }
    }
    
    return anlm;
    
}

vector2dc PairMLIPGtinv::compute_anlm_uniq_products
(const int& type1, const barray3dc& anlm){

    const int n_fn = pot.modelp.get_n_fn(); 
    const vector3i &type_comb_pair = pot.modelp.get_type_comb_pair();
    const vector2i &uniq_prod = pot.poly_gtinv.get_uniq_prod();
    const vector2i &lmtc_map = pot.poly_gtinv.get_lmtc_map();

    int lm, tc, type2;
    vector2dc prod(n_fn, vector1dc(uniq_prod.size(), 1.0));
    for (int i = 0; i < uniq_prod.size(); ++i){
        for (const auto &seq: uniq_prod[i]){
            lm = lmtc_map[seq][0], tc = lmtc_map[seq][1];
            if (type_comb_pair[tc][type1].size() > 0) {
                type2 = type_comb_pair[tc][type1][0];
                for (int n = 0; n < n_fn; ++n)
                    prod[n][i] *= anlm[type2][n][lm];
            }
            else {
                for (int n = 0; n < n_fn; ++n) prod[n][i] = 0.0;
                break;
            }
        }
    }
    return prod;
}

vector1d PairMLIPGtinv::compute_polynomial_model_uniq_products
(const int& type1, const barray3dc& anlm, const vector2dc& uniq){

    const int n_fn = pot.modelp.get_n_fn(); 
    const int n_des = pot.modelp.get_n_des(); 
    const int n_gtinv = pot.modelp.get_linear_term_gtinv().size();
    const int n_lm = pot.lm_info.size();
    const int n_lm_all = 2 * n_lm - pot.fp.maxl - 1;
    const int n_type = pot.fp.n_type;

    vector1d dn = vector1d(n_des, 0.0);
    for (int type2 = 0; type2 < n_type; ++type2){
        const int tc0 = type_comb[type1][type2];
        for (int lm0 = 0; lm0 < n_lm_all; ++lm0){
            for (const auto& t: pot.poly_gtinv.get_gtinv_info_poly(tc0, lm0)){
                if (t.lmt_pi == -1) {
                    for (int n = 0; n < n_fn; ++n){
                        dn[n * n_gtinv + t.reg_i] += anlm[type2][n][0].real();
                    }
                }
                else {
                    for (int n = 0; n < n_fn; ++n){
                        dn[n * n_gtinv + t.reg_i] += t.coeff / t.order 
                            * prod_real(anlm[type2][n][lm0],uniq[n][t.lmt_pi]);
                    }
                }
            }
        
        }
    }

    const auto &uniq_comb = pot.poly_gtinv.get_uniq_comb();
    vector1d uniq_prod(uniq_comb.size(), 0.5);
    for (int n = 0; n < uniq_comb.size(); ++n){
        for (const auto& c: uniq_comb[n]) uniq_prod[n] *= dn[c];
    }

    return uniq_prod;
}

double PairMLIPGtinv::prod_real(const dc& val1, const dc& val2){
    return val1.real() * val2.real() - val1.imag() * val2.imag();
}

/* ---------------------------------------------------------------------- */

void PairMLIPGtinv::allocate()
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

void PairMLIPGtinv::settings(int narg, char **arg)
{
 // force->newton_pair = 0;
  force->newton_pair = 1;
  if (narg != 0) error->all(FLERR,"Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairMLIPGtinv::coeff(int narg, char **arg)
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

double PairMLIPGtinv::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  return cutmax;
}

/* ---------------------------------------------------------------------- */

void PairMLIPGtinv::read_pot(char *file)
{
    std::ifstream input(file);
    if (input.fail()){
        std::cerr << "Error: Could not open mlip file: " << file << "\n";
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

    //struct DataMLIP pot;

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

    if (pot.fp.des_type != "gtinv"){
        error->all(FLERR,"des_type must be gtinv");
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
        pot.lm_info = get_lm_info(pot.fp.maxl);
    }

    // line 11: number of regression coefficients
    // line 12,13: regression coefficients, scale coefficients
    int n_reg_coeffs = get_value<int>(input);
    pot.reg_coeffs = get_value_array<double>(input, n_reg_coeffs);
    vector1d scale = get_value_array<double>(input, n_reg_coeffs);
    for (int i = 0; i < n_reg_coeffs; ++i) pot.reg_coeffs[i] *= 2.0/scale[i];

    // line 14: number of gaussian parameters
    // line 15-: gaussian parameters
    int n_params = get_value<int>(input);
    pot.fp.params = vector2d(n_params);
    for (int i = 0; i < n_params; ++i)
        pot.fp.params[i] = get_value_array<double>(input, 2);
    
    // last line: atomic mass
    mass = get_value_array<double>(input, ele.size());

    pot.modelp = ModelParams(pot.fp);
    pot.poly_gtinv = PolynomialGtinv(pot.fp, pot.modelp, pot.lm_info);

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
T PairMLIPGtinv::get_value(std::ifstream& input)
{
    std::string line;
    std::stringstream ss;

    T val;
    std::getline( input, line );
    ss << line;
    ss >> val;

    return val;
}

template<typename T>
std::vector<T> PairMLIPGtinv::get_value_array
(std::ifstream& input, const int& size)
{
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


