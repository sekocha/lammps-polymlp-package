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

#ifndef __MLIP_FEATURES
#define __MLIP_FEATURES

#include <boost/geometry.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>

#include "mlip_pymlcpp.h"
#include "mlip_basis_function.h"

namespace bg = boost::geometry;
namespace bm = boost::math;

// radial functions
void get_fn(const double& dis, const struct feature_params& fp, vector1d& fn);

void get_fn
(const double& dis, const struct feature_params& fp, 
 vector1d& fn, vector1d& fn_dr);

// Spherical harmonics
vector1d cartesian_to_spherical(const vector1d& v);

void get_ylm(const vector1d& sph, const vector2i& lm_info, vector1dc& ylm);

void get_ylm
(const vector1d& sph, const vector2i& lm_info,
 vector1dc& ylm, vector1dc& ylm_dtheta);

dc spherical_harmonic_dtheta
(const int& l, const int& m, const double& tan_theta,
 const dc& exp_imag_phi, const vector1dc& ylm, const int& lm);

vector2i get_lm_info(const int& max_l);

#endif
