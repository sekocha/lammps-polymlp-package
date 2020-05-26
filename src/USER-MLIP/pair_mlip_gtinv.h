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

#ifdef PAIR_CLASS

PairStyle(mlip_gtinv,PairMLIPGtinv)

#else

#ifndef LMP_PAIR_MLIP_GTINV_H
#define LMP_PAIR_MLIP_GTINV_H

#include "pair.h"

#include "mlip_pymlcpp.h"
#include "mlip_read_gtinv.h"
#include "mlip_features.h"
#include "mlip_model_params.h"
#include "mlip_polynomial.h"
#include "mlip_polynomial_gtinv.h"

namespace LAMMPS_NS {

class PairMLIPGtinv : public Pair {
 public:
  PairMLIPGtinv(class LAMMPS *);
  virtual ~PairMLIPGtinv();
  virtual void compute(int, int);
  void settings(int, char **);
  virtual void coeff(int, char **);
  
  virtual double init_one(int, int);
 /* virtual void init_style();
  */

 protected:

  virtual void allocate();

  struct DataMLIP {
      struct feature_params fp;
      ModelParams modelp;
      vector2i lm_info;
      PolynomialGtinv poly_gtinv;
      vector1d reg_coeffs;
  };

  std::vector<std::string> ele;
  double cutmax;
  vector1d mass;
  vector1i types;
  vector2i type_comb;

  struct DataMLIP pot;

  barray4dc compute_anlm();
  vector2dc compute_anlm_uniq_products
    (const int& type1, const barray3dc& anlm);
  vector1d compute_polynomial_model_uniq_products
    (const int& type1, const barray3dc& anlm, const vector2dc& prod);

//  vector1d polynomial_model_uniq_products(const vector1d& dn);
  void polynomial_sum
      (const vector1dc& uniq, const double& regc, 
       const double& coeff, const double& order, const double& lmt_pi, 
       dc& sume, dc& sumf);

  double prod_real(const dc& val1, const dc& val2);

  void read_pot(char *);
  template<typename T> T get_value(std::ifstream& input);
  template<typename T> std::vector<T> get_value_array
    (std::ifstream& input, const int& size);

};

}

#endif
#endif

