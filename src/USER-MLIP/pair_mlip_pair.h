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

PairStyle(mlip_pair,PairMLIPPair)

#else

#ifndef LMP_PAIR_MLIP_PAIR_H
#define LMP_PAIR_MLIP_PAIR_H

#include "pair.h"

#include "mlip_pymlcpp.h"
#include "mlip_read_gtinv.h"
#include "mlip_features.h"
#include "mlip_model_params.h"
#include "mlip_polynomial.h"
#include "mlip_polynomial_pair.h"

namespace LAMMPS_NS {

class PairMLIPPair : public Pair {
 public:
  PairMLIPPair(class LAMMPS *);
  virtual ~PairMLIPPair();
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
      PolynomialPair poly_model;
      vector1d reg_coeffs;
  };

  struct DataMLIP pot;

  std::vector<std::string> ele;
  double cutmax;
  vector1d mass;
  vector1i types;
  vector2i type_comb;

  vector1d polynomial_model_uniq_products(const vector1d& dn);
  double dot(const vector1d& a, const vector1d& b, const int& sindex);

  void read_pot(char *);
  template<typename T> T get_value(std::ifstream& input);
  template<typename T> std::vector<T> get_value_array
    (std::ifstream& input, const int& size);

};

}

#endif
#endif

