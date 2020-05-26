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

	    Header file for model_params.cpp
		
****************************************************************************/

#ifndef __MLIP_MODEL_PARAMS
#define __MLIP_MODEL_PARAMS

#include <set>
#include <iterator>
#include <algorithm>

#include "mlip_pymlcpp.h"

class ModelParams{

    int n_type, n_fn, n_des, n_coeff_all;
    vector2i comb2, comb3;
    vector3i type_comb_pair;

    std::vector<struct LinearTermGtinv> linear_array_g;

    void combination2(const vector1i& iarray, const std::string& des_type);
    void combination3(const vector1i& iarray, const std::string& des_type);
    int seq2typecomb(const int& i, const std::string& des_type);

    bool check_type_comb_pair(const vector1i& index, const int& type1) const;
    void uniq_gtinv_type(const feature_params& fp);

    public: 

    ModelParams();
    ModelParams(const struct feature_params& fp);
    ~ModelParams();

    const int& get_n_type() const;
    const int& get_n_fn() const;
    const int& get_n_des() const;
    const int& get_n_coeff_all() const;

    const vector2i& get_comb2() const;
    const vector2i& get_comb3() const;

    const std::vector<struct LinearTermGtinv>& get_linear_term_gtinv() const;

    const vector3i& get_type_comb_pair() const;
    vector1i get_type_comb_pair
        (const vector1i& tc_index, const int& type1);

};

template < typename SEQUENCE >
void Permutenr
(const SEQUENCE& input, SEQUENCE output, 
 std::vector<SEQUENCE>& all, std::size_t r){
    if( output.size() == r ) all.emplace_back(output); 
    else {
        for( std::size_t i=0; i < input.size(); ++i ) {
            SEQUENCE temp_output = output;
            temp_output.push_back(input[i]);
            Permutenr(input, temp_output, all, r);
        }
    }
}

struct LinearTermGtinv {
    int lmindex;
    vector1i tcomb_index;
    vector1i type1;
};

#endif
