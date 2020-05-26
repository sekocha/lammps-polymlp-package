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

*****************************************************************************/

#include "mlip_polynomial_pair.h"

PolynomialPair::PolynomialPair(){}
PolynomialPair::PolynomialPair
(const struct feature_params& fp, const ModelParams& modelp){

    const int n_tc = modelp.get_type_comb_pair().size();
    const int n_fn = modelp.get_n_fn();
    const int n_des = modelp.get_n_des();

    poly_array.resize(n_tc);
    for (int i = 0; i < n_tc; ++i) poly_array[i].resize(n_fn);

    poly_obj = Polynomial(fp, modelp, vector2i{});
    const auto &comb2 = poly_obj.get_comb2(), &comb3 = poly_obj.get_comb3();

    set_uniq_comb(comb2);
    set_uniq_comb(comb3);
    set_polynomial_array(comb2, n_des);
    set_polynomial_array(comb3, n_des + comb2.size());

    uniq_comb = vector2i(uniq_comb_set.begin(), uniq_comb_set.end());
}

PolynomialPair::~PolynomialPair(){}

void PolynomialPair::set_uniq_comb(const vector2i& comb_all){

    if (comb_all.size() > 0){
        const int order = comb_all[0].size();
        const vector2i &seq_array = poly_obj.permutation(order);

        vector1i comb_i_info(order-1);
        for (const auto& c: comb_all){
            for (const auto& seq: seq_array){
                for (int i = 1; i < order; ++i) comb_i_info[i-1] = c[seq[i]]; 
                uniq_comb_set.insert(comb_i_info);
            }
        }
    }
}
 
void PolynomialPair::set_polynomial_array
(const vector2i& comb_all, const int& sindex){

    if (comb_all.size() > 0){
        const int order = comb_all[0].size();
        const vector2i &seq_array = poly_obj.permutation(order);

        int tc0,n0,reg_i(sindex),cindex;
        vector1i comb_i_info(order-1);
        for (const auto& c: comb_all){
            for (const auto& seq: seq_array){
                for (int i = 1; i < order; ++i) comb_i_info[i-1] = c[seq[i]]; 
                cindex = poly_obj.find_comb(uniq_comb_set, comb_i_info, 0);
                poly_obj.seq2tcn(c[seq[0]], tc0, n0);
                struct PolynomialLammps pl = {n0, reg_i, cindex, order};
                poly_array[tc0][n0].emplace_back(pl);
            }
            ++reg_i;
        }
    }
}

const polyvec1& PolynomialPair::get_polynomial_info
(const int& tc, const int& n0) const{ 
    return poly_array[tc][n0]; 
}
const vector2i& PolynomialPair::get_uniq_comb() const { return uniq_comb; }


