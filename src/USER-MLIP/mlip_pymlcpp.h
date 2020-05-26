/****************************************************************************

        Copyright (C) 2018 Atsuto Seko
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

#ifndef __PYMLCPP_MODEL
#define __PYMLCPP_MODEL

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <map>
#include <array>
#include <string>
#include <complex>
#include <numeric>

#include "boost/multi_array.hpp"

using vector1i = std::vector<int>;
using vector2i = std::vector<vector1i>;
using vector3i = std::vector<vector2i>;
using vector4i = std::vector<vector3i>;
using vector1d = std::vector<double>;
using vector2d = std::vector<vector1d>;
using vector3d = std::vector<vector2d>;
using vector4d = std::vector<vector3d>;
using vector5d = std::vector<vector4d>;
using dc = std::complex<double>;
using vector1dc = std::vector<dc>;
using vector2dc = std::vector<vector1dc>;
using vector3dc = std::vector<vector2dc>;
using vector4dc = std::vector<vector3dc>;
using vector5dc = std::vector<vector4dc>;
using barray1d = boost::multi_array<double, 1>;
using barray2d = boost::multi_array<double, 2>;
using barray3d = boost::multi_array<double, 3>;
using barray4d = boost::multi_array<double, 4>;
using barray1dc = boost::multi_array<dc, 1>;
using barray2dc = boost::multi_array<dc, 2>;
using barray3dc = boost::multi_array<dc, 3>;
using barray4dc = boost::multi_array<dc, 4>;

template<typename T>
void print_time(clock_t& start, clock_t& end, const T& memo){

    std::cout << " elapsed time: " << memo << ": " 
        << (double)(end-start) / CLOCKS_PER_SEC << " (sec.)" << std::endl;

}

struct feature_params {
    int n_type;
    bool force;
    vector2d params;
    double cutoff;
    std::string pair_type;
    std::string des_type;
    int model_type;
    int maxp; 
    int maxl;
    vector3i lm_array;
    vector2i l_comb;
    vector2d lm_coeffs;
};

#endif
