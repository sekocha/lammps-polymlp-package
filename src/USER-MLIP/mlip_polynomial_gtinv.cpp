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

#include "mlip_polynomial_gtinv.h"

PolynomialGtinv::PolynomialGtinv(){}
PolynomialGtinv::PolynomialGtinv
(const struct feature_params& fp, const ModelParams& modelp, 
 const vector2i& lm_info){

    poly_obj = Polynomial(fp, modelp, lm_info);
    lmtc_map = poly_obj.get_lmtc_map();

    n_tc = modelp.get_type_comb_pair().size();
    n_fn = modelp.get_n_fn();
    n_des = modelp.get_n_des();
    n_lm_all = 2 * lm_info.size() - fp.maxl -1;

    set_uniq_lmt_prod(fp, modelp);
    set_gtinv(fp, modelp);
    uniq_lmt_prod_vec = vector2i(uniq_lmt_prod.begin(), uniq_lmt_prod.end());

    if (fp.maxp > 1){
        poly_array = vector4GPL(n_tc,vector3GPL(n_fn, vector2GPL(n_lm_all)));
        gtinv_array_poly = vector3GL(n_tc,vector2GL(n_lm_all));

        const auto &comb2 = poly_obj.get_comb2();
        const auto &comb3 = poly_obj.get_comb3();
        set_uniq_comb(comb2);
        set_uniq_comb(comb3);
        set_polynomial_array(comb2, n_des);
        set_polynomial_array(comb3, n_des + comb2.size());

        gtinv_extract();
        uniq_comb = vector2i(uniq_comb_set.begin(), uniq_comb_set.end());
    }
}

PolynomialGtinv::~PolynomialGtinv(){}

void PolynomialGtinv::set_uniq_lmt_prod
(const struct feature_params& fp, const ModelParams& modelp){

    const std::vector<LinearTermGtinv> &linear 
        = modelp.get_linear_term_gtinv();

    int seq, lmtc_seq;
    for (int i1 = 0; i1 < linear.size(); ++i1){
        const vector1i &tc = linear[i1].tcomb_index;
        const vector2i &lm_array = fp.lm_array[linear[i1].lmindex];

        const int order = tc.size();
        const vector2i &seq_array = poly_obj.permutation(order);
        for (int j = 0; j < order; ++j){
            for (int k1 = 0; k1 < lm_array.size(); ++k1){
                vector1i lmt_pi_info;
                for (int k2 = 1; k2 < lm_array[k1].size(); ++k2){
                    seq = seq_array[j][k2];
                    lmtc_seq = poly_obj.lmtc2seq(lm_array[k1][seq], tc[seq]);
                    lmt_pi_info.emplace_back(lmtc_seq);
                }
                std::sort(lmt_pi_info.begin(), lmt_pi_info.end());
                if (lmt_pi_info.size() > 0) 
                    uniq_lmt_prod.insert(lmt_pi_info);
            }
        }
    }
}

void PolynomialGtinv::set_gtinv
(const struct feature_params& fp, const ModelParams& modelp){

    const std::vector<LinearTermGtinv> &linear = modelp.get_linear_term_gtinv();
    gtinv_array = vector3GL(n_tc,vector2GL(n_lm_all));
    gtinv_all.resize(linear.size());

    int tc0,lm0,seq0,seq,lmtc_seq,reg_i,lmt_pi;
    for (int i1 = 0; i1 < linear.size(); ++i1){
        const vector1i &tc = linear[i1].tcomb_index;
        const vector2i &lm_array = fp.lm_array[linear[i1].lmindex];
        const vector1d &coeffs = fp.lm_coeffs[linear[i1].lmindex];

        reg_i = i1;
        const int order = tc.size();
        const vector2i &seq_array = poly_obj.permutation(order);
        for (int j = 0; j < order; ++j){
            seq0 = seq_array[j][0], tc0 = tc[seq0];
            for (int k1 = 0; k1 < lm_array.size(); ++k1){
                lm0 = lm_array[k1][seq0];
                vector1i lmt_pi_info;
                for (int k2 = 1; k2 < lm_array[k1].size(); ++k2){
                    seq = seq_array[j][k2];
                    lmtc_seq = poly_obj.lmtc2seq(lm_array[k1][seq], tc[seq]);
                    lmt_pi_info.emplace_back(lmtc_seq);
                }
                if (lmt_pi_info.size() > 0)
                    lmt_pi = poly_obj.find_comb(uniq_lmt_prod, lmt_pi_info, 0);
                else lmt_pi = -1;

                struct GtinvLammps gt = {reg_i,lmt_pi,coeffs[k1],order,tc0,lm0};
                gtinv_array[tc0][lm0].emplace_back(gt);
                gtinv_all[i1].emplace_back(gt);
            }
        }
    }
}

void PolynomialGtinv::set_uniq_comb(const vector2i& comb_all){

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

void PolynomialGtinv::set_polynomial_array
(const vector2i& comb_all, const int& sindex){

    if (comb_all.size() > 0){
        const int order = comb_all[0].size();
        const vector2i &seq_array = poly_obj.permutation(order);

        int cindex,c0,n0,igtinv0,reg_i(sindex);
        vector1i comb_i_info(order-1);
        for (const auto& c: comb_all){
            // order should be reconsidered
            int poly_order(0);
            for (const auto& ci: c){
                int ntmp, itmp;
                poly_obj.seq2ngtinv(ci, ntmp, itmp);
                poly_order += gtinv_all[itmp][0].order;
            }
            //////////////////////////////
            for (const auto& seq: seq_array){
                for (int i = 1; i < order; ++i) comb_i_info[i-1] = c[seq[i]];
                cindex = poly_obj.find_comb(uniq_comb_set, comb_i_info, 0);
                c0 = c[seq[0]];
                poly_obj.seq2ngtinv(c0, n0, igtinv0);
                for (const auto &gt: gtinv_all[igtinv0]){
                    // order should be reconsidered
                    // order * gt.order ?
                    struct GtinvPolynomialLammps pl = 
                        {c0, reg_i, cindex, gt.lmt_pi, gt.coeff, poly_order};
                    //////////////////////////////
                    poly_array[gt.tc0][n0][gt.lm0].emplace_back(pl);
                }

            }
            ++reg_i;
        }
    }
}

void PolynomialGtinv::gtinv_extract(){

    std::set<int> uniq_igtinv;
    for (const auto& uniq: uniq_comb_set){
        for (const auto& c: uniq){
            int ntmp, itmp;
            poly_obj.seq2ngtinv(c, ntmp, itmp);
            uniq_igtinv.insert(itmp);
        }
    }

    for (int tc0 = 0; tc0 < gtinv_array.size(); ++tc0){
        for (int lm0 = 0; lm0 < gtinv_array[tc0].size(); ++lm0){
            for (const auto& gt: gtinv_array[tc0][lm0]){
                if (uniq_igtinv.find(gt.reg_i) != uniq_igtinv.end())
                    gtinv_array_poly[tc0][lm0].emplace_back(gt);
            }
        }
    }
}


const vector2i& PolynomialGtinv::get_uniq_prod() const{ 
    return uniq_lmt_prod_vec; 
}
const vector2i& PolynomialGtinv::get_uniq_comb() const{ 
    return uniq_comb; 
}
const vector2i& PolynomialGtinv::get_lmtc_map() const { 
    return lmtc_map; 
}
const vector1GL& PolynomialGtinv::get_gtinv_info
(const int& tc0, const int& lm0) const{ 
    return gtinv_array[tc0][lm0]; 
}
const vector1GL& PolynomialGtinv::get_gtinv_info_poly
(const int& tc0, const int& lm0) const{ 
    return gtinv_array_poly[tc0][lm0]; 
}
const vector1GPL& PolynomialGtinv::get_polynomial_info
(const int& tc0, const int& n0, const int& lm0) const{ 
    return poly_array[tc0][n0][lm0]; 
}

