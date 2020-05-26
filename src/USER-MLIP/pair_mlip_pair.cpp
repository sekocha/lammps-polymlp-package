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

#include "pair_mlip_pair.h"

using namespace LAMMPS_NS;

#define MAXLINE 1024
#define DELTA 4

/* ---------------------------------------------------------------------- */

PairMLIPPair::PairMLIPPair(LAMMPS *lmp) : Pair(lmp)
{}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairMLIPPair::~PairMLIPPair()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

/* ---------------------------------------------------------------------- */

void PairMLIPPair::compute(int eflag, int vflag)
{

    vflag = 1;
    if (eflag || vflag) ev_setup(eflag,vflag);
    else evflag = 0;

    int inum = list->inum; 
    int nlocal = atom->nlocal;
    int newton_pair = force->newton_pair;

    // first part of polynomial model correction
    const int n_type_comb = pot.modelp.get_type_comb_pair().size();
    vector3d prod_all_f(n_type_comb, vector2d(inum));
    vector3d prod_all_e(n_type_comb, vector2d(inum));
    if (pot.fp.maxp > 1){
        vector2d dn(inum, vector1d(pot.modelp.get_n_des(), 0.0));
        #ifdef _OPENMP
        #pragma omp parallel for schedule(guided)
        #endif
        for (int ii = 0; ii < inum; ii++) {
            int i,j,type1,type2,jnum,sindex,*ilist,*jlist;
            double delx,dely,delz,dis;
            double **x = atom->x;
            tagint *tag = atom->tag;

            vector1d fn;
            const int n_fn = pot.modelp.get_n_fn();

            i = list->ilist[ii]; 
            type1 = types[tag[i]-1];
            jnum = list->numneigh[i]; 
            jlist = list->firstneigh[i];
            for (int jj = 0; jj < jnum; jj++) {
                j = jlist[jj]; 
                delx = x[i][0]-x[j][0];
                dely = x[i][1]-x[j][1];
                delz = x[i][2]-x[j][2];
                dis = sqrt(delx*delx + dely*dely + delz*delz);

                if (dis < pot.fp.cutoff){
                    type2 = types[tag[j]-1]; 
                    sindex = type_comb[type1][type2] * n_fn;
                    get_fn(dis, pot.fp, fn);
                    for (int n = 0; n < n_fn; ++n) {
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        dn[i][sindex+n] += fn[n];
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        dn[tag[j]-1][sindex+n] += fn[n];
                    }
                }
            }
        }
        
        #ifdef _OPENMP
        #pragma omp parallel for schedule(guided)
        #endif
        for (int ii = 0; ii < inum; ii++) {
            int i,*ilist,type1;
            double **x = atom->x;
            tagint *tag = atom->tag;
            ilist = list->ilist;

            i = ilist[ii], type1 = types[tag[i]-1];
            const int n_fn = pot.modelp.get_n_fn();
            const vector1d &prodi = polynomial_model_uniq_products(dn[i]);
            for (int type2 = 0; type2 < pot.fp.n_type; ++type2){
                const int tc = type_comb[type1][type2];
                vector1d vals_f(n_fn, 0.0), vals_e(n_fn, 0.0);
                for (int n = 0; n < n_fn; ++n){
                    double v;
                    for (const auto& pi: 
                        pot.poly_model.get_polynomial_info(tc,n)){
                        v = prodi[pi.comb_i] * pot.reg_coeffs[pi.reg_i];
                        vals_f[n] += v;
                        vals_e[n] += v / pi.order;
                    }
                }
                prod_all_f[tc][i] = vals_f;
                prod_all_e[tc][i] = vals_e;
            }
        }
    }
    // end: first part of polynomial model correction 

    vector2d evdwl_array(inum),fpair_array(inum);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided)
    #endif
    for (int ii = 0; ii < inum; ii++) {
        int i,j,jnum,*jlist,type1,type2,sindex,tc;
        double delx,dely,delz,dis,fpair,evdwl;
        double **x = atom->x;
        tagint *tag = atom->tag;

        i = list->ilist[ii]; 
        type1 = types[tag[i]-1];
        jnum = list->numneigh[i]; 
        jlist = list->firstneigh[i];

        const int n_fn = pot.modelp.get_n_fn();
        vector1d fn, fn_d;

        evdwl_array[ii].resize(jnum);
        fpair_array[ii].resize(jnum);
        for (int jj = 0; jj < jnum; jj++) {
            j = jlist[jj]; 
            delx = x[i][0]-x[j][0];
            dely = x[i][1]-x[j][1];
            delz = x[i][2]-x[j][2];
            dis = sqrt(delx*delx + dely*dely + delz*delz);

            if (dis < pot.fp.cutoff){
                type2 = types[tag[j]-1]; 
                tc = type_comb[type1][type2];
                sindex = tc * n_fn;

                get_fn(dis, pot.fp, fn, fn_d);
                fpair = dot(fn_d, pot.reg_coeffs, sindex);
                evdwl = dot(fn, pot.reg_coeffs, sindex);

                // polynomial model correction
                if (pot.fp.maxp > 1){
                    fpair += dot(fn_d, prod_all_f[tc][i], 0)
                        +  dot(fn_d, prod_all_f[tc][tag[j]-1], 0);
                    evdwl += dot(fn, prod_all_e[tc][i], 0)
                        + dot(fn, prod_all_e[tc][tag[j]-1], 0);
                }
                // polynomial model correction: end

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

vector1d PairMLIPPair::polynomial_model_uniq_products(const vector1d& dn){

    const auto &uniq_comb = pot.poly_model.get_uniq_comb();
    vector1d prod(uniq_comb.size(), 0.5);
    for (int n = 0; n < uniq_comb.size(); ++n){
        for (const auto& c: uniq_comb[n]) prod[n] *= dn[c];
    }

    return prod;
}

double PairMLIPPair::dot
(const vector1d& a, const vector1d& b, const int& sindex){
    double val(0.0);
    for (int n = 0; n < a.size(); ++n) val += a[n] * b[sindex+n];
    return val;
}

/* ---------------------------------------------------------------------- */

void PairMLIPPair::allocate()
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

void PairMLIPPair::settings(int narg, char **arg)
{
  force->newton_pair = 1;
  if (narg != 0) error->all(FLERR,"Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairMLIPPair::coeff(int narg, char **arg)
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

double PairMLIPPair::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  return cutmax;
}

/* ---------------------------------------------------------------------- */

void PairMLIPPair::read_pot(char *file)
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

    if (pot.fp.des_type != "pair"){
        error->all(FLERR,"des_type must be pair");
    }

    // line 11: number of regression coefficients
    // line 12,13: regression coefficients, scale coefficients
    int n_reg_coeffs = get_value<int>(input);
    pot.reg_coeffs = get_value_array<double>(input, n_reg_coeffs);
    vector1d scale = get_value_array<double>(input, n_reg_coeffs);
    for (int i = 0; i < n_reg_coeffs; ++i) 
        pot.reg_coeffs[i] *= 2.0/scale[i];

    // line 14: number of gaussian parameters
    // line 15-: gaussian parameters
    int n_params = get_value<int>(input);
    pot.fp.params = vector2d(n_params);
    for (int i = 0; i < n_params; ++i)
        pot.fp.params[i] = get_value_array<double>(input, 2);
    
    // last line: atomic mass
    mass = get_value_array<double>(input, ele.size());

    pot.modelp = ModelParams(pot.fp);
    if (pot.fp.maxp > 1) 
        pot.poly_model = PolynomialPair(pot.fp, pot.modelp);

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
T PairMLIPPair::get_value(std::ifstream& input)
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
std::vector<T> PairMLIPPair::get_value_array
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
