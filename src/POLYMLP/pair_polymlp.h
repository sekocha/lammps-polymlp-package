/* -*- c++ -*- ----------------------------------------------------------
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

#ifdef PAIR_CLASS

PairStyle(polymlp,PairPolyMLP)

#else

#ifndef LMP_PAIR_POLYMLP_H
#define LMP_PAIR_POLYMLP_H

#include "pair.h"

#include "polymlp_mlpcpp.h"
#include "polymlp_read_gtinv.h"
#include "polymlp_functions_interface.h"
#include "polymlp_model_params.h"

#include "polymlp_features.h"
#include "polymlp_potential.h"

namespace LAMMPS_NS {

class PairPolyMLP : public Pair {
 public:
  PairPolyMLP(class LAMMPS *);
  virtual ~PairPolyMLP();
  virtual void compute(int, int);
  void settings(int, char **);
  virtual void coeff(int, char **);
  
  virtual double init_one(int, int);
 /* virtual void init_style();
  */

 protected:

  virtual void allocate();

  struct DataPolyMLP {
      struct feature_params fp;
      ModelParams modelp;
      Potential p_obj;
  };

  std::vector<std::string> ele;
  double cutmax;
  vector1d mass;
  vector1i types;
  vector2i type_comb;
  vector2i nlmtc_map_no_conj_key_array;

  struct DataPolyMLP pot;

  void compute_pair(int eflag, int vflag);
  void compute_gtinv(int eflag, int vflag);

  // for pair
  void compute_antc(vector2d& antc);
  void compute_sum_of_prod_antc(const vector2d& antc,
                                vector2d& prod_sum_e,
                                vector2d& prod_sum_f);

  // for gtinv
  void compute_anlmtc(vector2dc& anlmtc);
  void compute_anlmtc_openmp(vector2dc& anlmtc);
  void compute_anlmtc_conjugate(const vector2d& anlmtc_r, 
                                const vector2d& anlmtc_i, 
                                vector2dc& anlmtc);
  void compute_sum_of_prod_anlmtc(const vector2dc& anlmtc,
                                  vector2dc& prod_sum_e,
                                  vector2dc& prod_sum_f);

  void compute_linear_features(const vector1d& prod_anlmtc,
                               const int type1,
                               vector1d& feature_values);
  template<typename T>
  void compute_products(const vector2i& map, 
                        const std::vector<T>& element,
                        std::vector<T>& prod_vals);

  void compute_products_real(const vector2i& map, 
                             const vector1dc& element,
                             vector1d& prod_vals);

  double prod_real(const dc& val1, const dc& val2);
  dc prod_real_and_complex(const double val1, const dc& val2);

  void read_pot(char *);
  template<typename T> T get_value(std::ifstream& input);

  template<typename T>
    T get_value(std::ifstream& input, std::string& tag);
  template<typename T> std::vector<T> 
    get_value_array(std::ifstream& input, const int& size);

};

}

#endif
#endif

