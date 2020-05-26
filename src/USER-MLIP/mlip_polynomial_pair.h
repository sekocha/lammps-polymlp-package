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

#ifndef __MLIP_POLYNOMIAL_PAIR
#define __MLIP_POLYNOMIAL_PAIR

#include "mlip_pymlcpp.h"
#include "mlip_model_params.h"
#include "mlip_polynomial.h"

class PolynomialPair{

    vector2i uniq_comb;
    std::set<vector1i> uniq_comb_set;

    Polynomial poly_obj;
    polyvec3 poly_array;

    void set_uniq_comb(const vector2i& comb_all);
    void set_polynomial_array(const vector2i& combs, const int& sindex);

    public: 

    PolynomialPair();
    PolynomialPair(const struct feature_params& fp, const ModelParams& modelp);
    ~PolynomialPair();

    const polyvec1& get_polynomial_info(const int& tc, const int& n0) const;
    const vector2i& get_uniq_comb() const;

};

#endif
