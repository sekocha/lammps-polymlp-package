/****************************************************************************

        Copyright (C) 2020 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public License
        as published by the Free Software Foundation; either version 2
        of the License, or (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program; if not, write to
        the Free Software Foundation, Inc., 51 Franklin Street,
        Fifth Floor, Boston, MA 02110-1301, USA, or see
        http://www.gnu.org/copyleft/gpl.txt

****************************************************************************/

#ifndef __MLIP_POLYNOMIAL
#define __MLIP_POLYNOMIAL

#include <set>
#include <map>
#include <iterator>
#include <algorithm>

#include "mlip_pymlcpp.h"
#include "mlip_model_params.h"

typedef std::vector<struct PolynomialLammps> polyvec1;
typedef std::vector<polyvec1> polyvec2;
typedef std::vector<polyvec2> polyvec3;
typedef std::vector<struct GtinvLammps> vector1GL;
typedef std::vector<vector1GL> vector2GL;
typedef std::vector<vector2GL> vector3GL;
typedef std::vector<struct GtinvPolynomialLammps> vector1GPL;
typedef std::vector<vector1GPL> vector2GPL;
typedef std::vector<vector2GPL> vector3GPL;
typedef std::vector<vector3GPL> vector4GPL;

struct PolynomialLammps {
    int nindex0;
    int reg_i;
    int comb_i;
    int order;
};

struct GtinvLammps {
    int reg_i;
    int lmt_pi;
    double coeff;
    int order;
    int tc0;
    int lm0;
};

struct GtinvPolynomialLammps {
    int c0;
    int reg_i;
    int comb_i;
    int lmt_pi;
    double coeff;
    int order;
};

class Polynomial{

    int n_tc, n_fn, n_lm_all, n_gtinv;
    vector2i swap_rule, swap_map_poly, swap_map_lmtc, swap_map_gtinv, lmtc_map;
    vector2i comb2,comb3;

    void set_swap_map_tcn();
    void set_swap_map_lmtc();
    void set_swap_map_gtinv
        (const struct feature_params fp, const ModelParams& modelp);
    void set_swap_map_ngtinv();
    void set_lmtc_map();

    vector1i tc_array_swap
        (const int& tc0, const vector1i& tc_array, const vector1i& lc_array);

    public: 

    Polynomial();
    Polynomial
        (const struct feature_params& fp, const ModelParams& modelp, 
         const vector2i& lm_info);
    ~Polynomial();

    int find_comb
        (const std::set<vector1i>& uniq_comb_set,
         const vector1i& comb_i, const int& s);
    vector2i permutation(const int& order); 

    int tcn2seq(const int& tc, const int& n);
    int lmtc2seq(const int& lm, const int& tc);
    int ngtinv2seq(const int& n, const int& igtinv);

    void seq2tcn(const int& seq, int& tc, int& n);
    void seq2lmtc(const int& seq, int& lm, int& tc);
    void seq2ngtinv(const int& seq, int& n, int& igtinv);

    const vector2i& get_swap_map_poly() const;
    const vector2i& get_swap_map_lmtc() const;
    const vector2i& get_swap_map_gtinv() const;
    const vector2i& get_lmtc_map() const;

    const vector2i& get_comb2() const;
    const vector2i& get_comb3() const;
};

#endif
