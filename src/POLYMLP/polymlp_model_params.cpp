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

#include "polymlp_model_params.h"

ModelParams::ModelParams(){}
ModelParams::ModelParams(const struct feature_params& fp){

    n_type = fp.n_type;

    // must be extended to more than ternary systems
    // Setting two elements in one array such as {0,1}, ... is unavailable.
    // If using such a setting, codes for computing features should be revised.
    if (n_type == 1) type_comb_pair = {{{0}}};
    else if (n_type == 2) type_comb_pair = {{{0}, {}}, {{1}, {0}}, {{}, {1}}};

    n_type_comb = type_comb_pair.size();
    initial_setting(fp);
}

ModelParams::ModelParams(const struct feature_params& fp, const bool icharge){

    n_type = fp.n_type;

    if (icharge == false){
        if (n_type == 1) {
            type_comb_pair = {{{0}}};
        }
        else if (n_type == 2) {
            type_comb_pair = {{{0}, {}}, 
                              {{1}, {0}}, 
                              {{}, {1}}};
        }
        else if (n_type == 3) {
            type_comb_pair = {{{0}, {}, {}}, 
                              {{1}, {0}, {}}, 
                              {{2}, {}, {0}}, 
                              {{}, {1}, {}}, 
                              {{}, {2}, {1}}, 
                              {{}, {}, {2}}};
        }
        else if (n_type == 4) {
            type_comb_pair = {{{0}, {}, {}, {}}, 
                              {{1}, {0}, {}, {}}, 
                              {{2}, {}, {0}, {}}, 
                              {{3}, {}, {}, {0}}, 
                              {{}, {1}, {}, {}}, 
                              {{}, {2}, {1}, {}}, 
                              {{}, {3}, {}, {1}}, 
                              {{}, {}, {2}, {}}, 
                              {{}, {}, {3}, {2}}, 
                              {{}, {}, {}, {3}}};
        }
        else if (n_type == 5) {
            type_comb_pair = {{{0}, {}, {}, {}, {}}, 
                              {{1}, {0}, {}, {}, {}}, 
                              {{2}, {}, {0}, {}, {}}, 
                              {{3}, {}, {}, {0}, {}}, 
                              {{4}, {}, {}, {}, {0}}, 
                              {{}, {1}, {}, {}, {}}, 
                              {{}, {2}, {1}, {}, {}}, 
                              {{}, {3}, {}, {1}, {}}, 
                              {{}, {4}, {}, {}, {1}}, 
                              {{}, {}, {2}, {}, {}}, 
                              {{}, {}, {3}, {2}, {}}, 
                              {{}, {}, {4}, {}, {2}}, 
                              {{}, {}, {}, {3}, {}},
                              {{}, {}, {}, {4}, {3}},
                              {{}, {}, {}, {}, {4}}};
        }
        else {
            exit(8);
        }
    }
    else {
    // Setting two elements in one array such as {0,1}, ... is unavailable.
    // If using such a setting, codes for computing features should be revised.
        if (n_type == 2) {
            type_comb_pair = {{{0}, {}},
                              {{1}, {}},
                              {{}, {0}},
                              {{}, {1}}};
        }
        else if (n_type == 3) {
            type_comb_pair = {{{0}, {}, {}}, 
                              {{1}, {}, {}}, 
                              {{2}, {}, {}}, 
                              {{}, {0}, {}}, 
                              {{}, {1}, {}}, 
                              {{}, {2}, {}}, 
                              {{}, {}, {0}}, 
                              {{}, {}, {1}}, 
                              {{}, {}, {2}}};
        }
        else {
            exit(8);
        }
    }

    n_type_comb = type_comb_pair.size();
    initial_setting(fp);
}

ModelParams::~ModelParams(){}

void ModelParams::initial_setting(const struct feature_params& fp){

    n_type = fp.n_type, n_fn = fp.params.size();

    if (fp.des_type == "pair") n_des = n_fn * type_comb_pair.size();
    else if (fp.des_type == "gtinv"){
        uniq_gtinv_type(fp);
        n_des = n_fn * linear_array_g.size();
    }

    vector1i polynomial_index;
    if (fp.model_type == 2){
        for (int n = 0; n < n_des; ++n) 
            polynomial_index.emplace_back(n);
    }
    else if (fp.model_type == 3 and fp.des_type == "gtinv"){
        for (int i = 0; i < linear_array_g.size(); ++i){
            const auto& lin = linear_array_g[i];
            if (lin.tcomb_index.size() == 1){
                for (int n = 0; n < n_fn; ++n){
                    polynomial_index.emplace_back(n*linear_array_g.size()+i);
                }
            }
        }
        std::sort(polynomial_index.begin(),polynomial_index.end());
    }
    else if (fp.model_type == 4 and fp.des_type == "gtinv"){
        for (int i = 0; i < linear_array_g.size(); ++i){
            const auto& lin = linear_array_g[i];
            if (lin.tcomb_index.size() < 3 ){
                for (int n = 0; n < n_fn; ++n){
                    polynomial_index.emplace_back(n*linear_array_g.size()+i);
                }
            }
        }
        std::sort(polynomial_index.begin(),polynomial_index.end());
    }


    if (fp.model_type == 1) n_coeff_all = n_des * fp.maxp;
    else if (fp.model_type > 1){
        if (fp.des_type == "pair"){
            if (fp.maxp > 1) combination2(polynomial_index);
            if (fp.maxp > 2) combination3(polynomial_index);
        }
        else if (fp.des_type == "gtinv"){
            if (fp.maxp > 1) combination2_gtinv(polynomial_index);
            if (fp.maxp > 2) combination3_gtinv(polynomial_index);
        }
        n_coeff_all = n_des + comb2.size() + comb3.size();
    }
}

void ModelParams::uniq_gtinv_type(const feature_params& fp){

    const vector2i &l_comb = fp.l_comb;
    vector1i pinput(type_comb_pair.size());
    for (int i = 0; i < type_comb_pair.size(); ++i) pinput[i] = i;

    const int gtinv_order = (*(l_comb.end()-1)).size();
    vector3i perm_array(gtinv_order);
    for (int i = 0; i < gtinv_order; ++i){
        vector2i perm;
        Permutenr(pinput, vector1i({}), perm, i+1);
        for (const auto& p1: perm){
            for (int type1 = 0; type1 < n_type; ++type1){
                if (check_type_comb_pair(p1, type1) == true){
                    perm_array[i].emplace_back(p1);
                    break;
                }
            }
        }
    }

    for (int i = 0; i < l_comb.size(); ++i){
        const vector1i& lc = l_comb[i];
        std::set<std::multiset<std::pair<int,int> > > uniq_lmt;
        for (const auto &p: perm_array[lc.size()-1]){
            std::multiset<std::pair<int, int> > tmp;
            for (int j = 0; j < p.size(); ++j){
                tmp.insert(std::make_pair(lc[j], p[j]));
            }
            uniq_lmt.insert(tmp);
        }
        for (const auto& lt: uniq_lmt){
            vector1i tc, t1a;
            for (const auto& lt1: lt) tc.emplace_back(lt1.second);
            for (int type1 = 0; type1 < n_type; ++type1){
                if (check_type_comb_pair(tc, type1) == true) 
                    t1a.emplace_back(type1);
            }
            linear_array_g.emplace_back(LinearTermGtinv({i,tc,t1a}));
        }
    }
}

void ModelParams::combination2_gtinv(const vector1i& iarray){

    for (int i1 = 0; i1 < iarray.size(); ++i1){
        int t1 = seq2igtinv(iarray[i1]);
        const auto &type1_1 = linear_array_g[t1].type1;
        for (int i2 = 0; i2 <= i1; ++i2){
            int t2 = seq2igtinv(iarray[i2]);
            const auto &type1_2 = linear_array_g[t2].type1;
            if (check_type(vector2i{type1_1,type1_2}) == true)
                comb2.push_back(vector1i({iarray[i2],iarray[i1]}));
        }
    }
}
void ModelParams::combination3_gtinv(const vector1i& iarray){

    for (int i1 = 0; i1 < iarray.size(); ++i1){
        int t1 = seq2igtinv(iarray[i1]);
        const auto &type1_1 = linear_array_g[t1].type1;
        for (int i2 = 0; i2 <= i1; ++i2){
            int t2 = seq2igtinv(iarray[i2]);
            const auto &type1_2 = linear_array_g[t2].type1;
            for (int i3 = 0; i3 <= i2; ++i3){
                int t3 = seq2igtinv(iarray[i3]);
                const auto &type1_3 = linear_array_g[t3].type1;
                if (check_type(vector2i{type1_1,type1_2,type1_3}) == true)
                    comb3.push_back
                        (vector1i({iarray[i3],iarray[i2],iarray[i1]}));
            }
        }
    }
}


bool ModelParams::check_type(const vector2i &type1_array){

    for (int type1 = 0; type1 < n_type; ++type1){
        bool tag = true;
        for (const auto &t1: type1_array){
            if (std::find(t1.begin(),t1.end(),type1) == t1.end()){
                tag = false;
                break;
            }
        }
        if (tag == true) return true;
    }
    return false;
}

int ModelParams::seq2typecomb(const int& seq){ 
    return seq/n_fn;
}
int ModelParams::seq2igtinv(const int& seq){
    return seq % linear_array_g.size();
}

void ModelParams::combination2(const vector1i& iarray){

    for (int i1 = 0; i1 < iarray.size(); ++i1){
        int t1 = seq2typecomb(iarray[i1]);
        for (int i2 = 0; i2 <= i1; ++i2){
            int t2 = seq2typecomb(iarray[i2]);
            for (int type1 = 0; type1 < n_type; ++type1){
                if (check_type_comb_pair(vector1i({t1,t2}), type1) == true){
                    comb2.push_back(vector1i({iarray[i2],iarray[i1]}));
                    break;
                }
            }
        }
    }
}

void ModelParams::combination3(const vector1i& iarray){

    for (int i1 = 0; i1 < iarray.size(); ++i1){
        int t1 = seq2typecomb(iarray[i1]);
        for (int i2 = 0; i2 <= i1; ++i2){
            int t2 = seq2typecomb(iarray[i2]);
            for (int i3 = 0; i3 <= i2; ++i3){
                int t3 = seq2typecomb(iarray[i3]);
                for (int type1 = 0; type1 < n_type; ++type1){
                    if (check_type_comb_pair
                        (vector1i({t1,t2,t3}), type1) == true){
                        comb3.push_back
                            (vector1i({iarray[i3],iarray[i2],iarray[i1]}));
                        break;
                    }
                }
            }
        }
    }
}


bool ModelParams::check_type_comb_pair
(const vector1i& index, const int& type1) const{ 
    vector1i size;
    for (const auto& p2: index){
        size.emplace_back(type_comb_pair[p2][type1].size());
    }
    int minsize = *std::min_element(size.begin(), size.end());
    return minsize > 0;
}

const int& ModelParams::get_n_type() const { return n_type; }
const int& ModelParams::get_n_type_comb() const { return n_type_comb; }
const int& ModelParams::get_n_fn() const { return n_fn; }
const int& ModelParams::get_n_des() const { return n_des; }
const int& ModelParams::get_n_coeff_all() const { return n_coeff_all; }
const vector2i& ModelParams::get_comb2() const { return comb2; }
const vector2i& ModelParams::get_comb3() const{ return comb3; }

const vector3i& ModelParams::get_type_comb_pair() const{ 
    return type_comb_pair;
}

vector1i ModelParams::get_type_comb_pair(const vector1i& tc_index, 
                                         const int& type1){ 
    vector1i all;
    for (const auto& i: tc_index) 
        all.emplace_back(type_comb_pair[i][type1][0]);
    return all;
}

const std::vector<struct LinearTermGtinv>& 
ModelParams::get_linear_term_gtinv() const{
    return linear_array_g;
}

