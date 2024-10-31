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
#include "polymlp_parse_potential.h"
#include "polymlp_read_gtinv.h"
#include "polymlp_functions_interface.h"
#include "polymlp_products.h"

#include "polymlp_mapping.h"
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
    Mapping mapping;
    ModelParams modelp;
    Potential p_obj;
  };

  std::vector<std::string> ele;
  double cutmax;
  vector1d mass;
  vector1i types;
  vector2i type_pairs;
  //vector2i nlmtp_map_no_conj_key_array;

  struct DataPolyMLP pot;

  void compute_pair(int eflag, int vflag);
  void compute_gtinv(int eflag, int vflag);

  // for pair
  void compute_antp(vector2d& antp);
  void compute_sum_of_prod_antp(
    const vector2d& antp, vector2d& prod_sum_e, vector2d& prod_sum_f
  );

  // for gtinv
  void compute_anlmtp(vector2dc& anlmtp);
  void compute_anlmtp_openmp(vector2dc& anlmtp);
  void compute_anlmtp_conjugate(
    const vector2d& anlmtp_r, const vector2d& anlmtp_i, vector2dc& anlmtp
  );
  void compute_sum_of_prod_anlmtp(
    const vector2dc& anlmtp, vector2dc& prod_sum_e, vector2dc& prod_sum_f
  );

  void compute_linear_features(
    const vector1d& prod_anlmtp, const int type1, vector1d& feature_values
  );

  void read_pot(char *);

};

}

#endif
#endif

