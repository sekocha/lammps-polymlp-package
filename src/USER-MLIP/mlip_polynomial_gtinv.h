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

#ifndef __MLIP_POLYNOMIAL_GTINV
#define __MLIP_POLYNOMIAL_GTINV

#include <set>
#include <map>
#include <iterator>
#include <algorithm>

#include "mlip_pymlcpp.h"
#include "mlip_model_params.h"
#include "mlip_polynomial.h"

class PolynomialGtinv{

    Polynomial poly_obj;

    int n_tc, n_fn, n_des, n_lm_all;
    vector2i uniq_lmt_prod_vec, uniq_comb, lmtc_map;
    std::set<vector1i> uniq_lmt_prod, uniq_comb_set;

    vector3GL gtinv_array, gtinv_array_poly;
    vector2GL gtinv_all;
    vector4GPL poly_array;

    void set_uniq_lmt_prod
        (const struct feature_params& fp, const ModelParams& modelp);
    void set_gtinv(const struct feature_params& fp, const ModelParams& modelp);

    void set_uniq_comb(const vector2i& comb_all);
    void set_polynomial_array(const vector2i& comb_all, const int& sindex);
    void gtinv_extract();

    public: 

    PolynomialGtinv();
    PolynomialGtinv
        (const struct feature_params& fp, const ModelParams& modelp,
         const vector2i& lm_info);
    ~PolynomialGtinv();

    const vector1GL& get_gtinv_info
        (const int& tc0, const int& lm0) const;
    const vector1GL& get_gtinv_info_poly
        (const int& tc0, const int& lm0) const;
    const vector1GPL& get_polynomial_info
        (const int& tc0, const int& n0, const int& lm0) const;
    const vector2i& get_uniq_prod() const;
    const vector2i& get_uniq_comb() const;
    const vector2i& get_lmtc_map() const;
};

#endif
