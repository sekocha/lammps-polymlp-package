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

#include "mlip_polynomial.h"

Polynomial::Polynomial(){}
Polynomial::Polynomial
(const struct feature_params& fp, const ModelParams& modelp, 
 const vector2i& lm_info){

    if (fp.n_type == 1) swap_rule = {{0}};
    else if (fp.n_type == 2) swap_rule = {{0,1,2},{2,1,0},{0,1,2}};

    if (fp.model_type == 1){
        for (int n = 0; n < modelp.get_n_des(); ++n){
            if (fp.maxp > 1) comb2.emplace_back(vector1i{n,n});
            if (fp.maxp > 2) comb3.emplace_back(vector1i{n,n,n});
        }
    }
    else if (fp.model_type > 1){
        comb2 = modelp.get_comb2();
        comb3 = modelp.get_comb3();
    }

    n_tc = modelp.get_type_comb_pair().size();
    n_fn = modelp.get_n_fn();

    if (fp.des_type == "gtinv"){
        n_lm_all = 2 * lm_info.size() - fp.maxl -1;
        n_gtinv = modelp.get_linear_term_gtinv().size();
        set_lmtc_map();
//        set_swap_map_lmtc();
//        set_swap_map_gtinv(fp, modelp);
//        if (fp.maxp > 1) set_swap_map_ngtinv();
    }
}

Polynomial::~Polynomial(){}

void Polynomial::set_swap_map_tcn(){

    swap_map_poly = vector2i(n_tc, vector1i(n_tc*n_fn));
    int seq, tc_swap;
    for (int tc0 = 0; tc0 < n_tc; ++tc0){
        for (int tc1 = 0; tc1 < n_tc; ++tc1){
            for (int n = 0; n < n_fn; ++n){
                seq = tcn2seq(tc1, n);
                tc_swap = swap_rule[tc0][tc1];
                swap_map_poly[tc0][seq] = tcn2seq(tc_swap, n);
            }
        }
    }
}

void Polynomial::set_swap_map_lmtc(){

    swap_map_lmtc = vector2i(n_tc, vector1i(n_lm_all * n_tc));
    int seq, tc_swap;
    for (int tc0 = 0; tc0 < n_tc; ++tc0){
        for (int lm = 0; lm < n_lm_all; ++lm){
            for (int tc1 = 0; tc1 < n_tc; ++tc1){
                seq = lmtc2seq(lm, tc1);
                tc_swap = swap_rule[tc0][tc1];
                swap_map_lmtc[tc0][seq] = lmtc2seq(lm, tc_swap);
            }
        }
    }
}

void Polynomial::set_swap_map_gtinv
(const struct feature_params fp, const ModelParams& modelp){

    swap_map_gtinv = vector2i(n_tc, vector1i(n_gtinv));
    const auto &linear = modelp.get_linear_term_gtinv();
    for (int tc0 = 0; tc0 < n_tc; ++tc0){
        for (int seq = 0; seq < n_gtinv; ++seq){
            const vector1i &tc_swap = tc_array_swap
                (tc0, linear[seq].tcomb_index, fp.l_comb[linear[seq].lmindex]);
            for (int i2 = 0; i2 < n_gtinv; ++i2){
                if (linear[i2].lmindex == linear[seq].lmindex and 
                    linear[i2].tcomb_index == tc_swap){
                    swap_map_gtinv[tc0][seq] = i2;
                    break;
                }
            }
        }
    }
}

void Polynomial::set_swap_map_ngtinv(){

    swap_map_poly = vector2i(n_tc, vector1i(n_fn * n_gtinv));
    int seq, igtinv_swap;
    for (int tc0 = 0; tc0 < n_tc; ++tc0){
        for (int n = 0; n < n_fn; ++n){
            for (int igtinv = 0; igtinv < n_gtinv; ++igtinv){
                seq = ngtinv2seq(n, igtinv);
                igtinv_swap = swap_map_gtinv[tc0][igtinv];
                swap_map_poly[tc0][seq] = ngtinv2seq(n, igtinv_swap);
            }
        }
    }
}

vector1i Polynomial::tc_array_swap
(const int& tc0, const vector1i& tc_array, const vector1i& lc_array){

    vector1i tc_swap;
    for (const auto& tc1: tc_array) tc_swap.emplace_back(swap_rule[tc0][tc1]);

    std::multiset<std::pair<int, int> > ltc_j;
    for (int i = 0; i < lc_array.size(); ++i)
        ltc_j.insert(std::make_pair(lc_array[i],tc_swap[i]));

    vector1i tcj;
    for (const auto &v: ltc_j) tcj.emplace_back(v.second);

    return tcj;
}

int Polynomial::find_comb
(const std::set<vector1i>& uniq_comb_set,
 const vector1i& comb_i, const int& s){

    vector1i vec1(comb_i.begin() + s, comb_i.end());
    std::sort(vec1.begin(), vec1.end());
    auto itr_i = uniq_comb_set.find(vec1);
    int index = std::distance(uniq_comb_set.begin(), itr_i);

    return index;
}

vector2i Polynomial::permutation(const int& order){
    vector2i seq_array(order);
    for (int i = 0; i < order; ++i){
        seq_array[i].emplace_back(i);
        for (int j = 0; j < order; ++j){
            if (j != i) seq_array[i].emplace_back(j);
        }
    }
    return seq_array;
}



/*
int Polynomial::comb_find(const vector2i& vec_array, const vector1i& vec){

    int index(-1);
    vector1i vec_copy = vec;
    std::sort(vec_copy.begin(),vec_copy.end());
    for (int i = 0; i < vec_array.size(); ++i){
        if (vec_array[i] == vec_copy){
            index = i;
            break;
        }
    }
    return index;
}
*/

void Polynomial::set_lmtc_map(){ 
    
    for (int i = 0; i < n_lm_all; ++i){
        for (int j = 0; j < n_tc; ++j){
            lmtc_map.emplace_back(vector1i{i,j});
        }
    }
}

int Polynomial::tcn2seq(const int& tc, const int& n){
    return tc * n_fn + n;
}
int Polynomial::lmtc2seq(const int& lm, const int& tc){
    return lm * n_tc + tc;
}
int Polynomial::ngtinv2seq(const int& n, const int& igtinv){
    return n * n_gtinv + igtinv;
}

void Polynomial::seq2tcn(const int& seq, int& tc, int& n){
    tc = seq/n_fn, n = seq%n_fn;
}
void Polynomial::seq2lmtc(const int& seq, int& lm, int& tc){
    lm = seq/n_tc, tc = seq%n_tc;
}
void Polynomial::seq2ngtinv(const int& seq, int& n, int& igtinv){
    n = seq/n_gtinv, igtinv = seq%n_gtinv;
}
const vector2i& Polynomial::get_swap_map_poly() const { 
    return swap_map_poly; 
}
const vector2i& Polynomial::get_swap_map_lmtc() const { 
    return swap_map_lmtc; 
}
const vector2i& Polynomial::get_swap_map_gtinv() const { 
    return swap_map_gtinv; 
}
const vector2i& Polynomial::get_lmtc_map() const { 
    return lmtc_map; 
}
const vector2i& Polynomial::get_comb2() const { 
    return comb2; 
}
const vector2i& Polynomial::get_comb3() const { 
    return comb3; 
}
