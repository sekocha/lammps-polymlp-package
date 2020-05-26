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

#include "mlip_features.h"

void get_fn(const double& dis, const struct feature_params& fp, vector1d& fn){

    double fc = cosine_cutoff_function(dis, fp.cutoff);

    fn.resize(fp.params.size());
    if (fp.pair_type == "gaussian"){
        for (int n = 0; n < fp.params.size(); ++n){
            fn[n] = gauss(dis, fp.params[n][0], fp.params[n][1]) * fc;
        }
    }
}


void get_fn
(const double& dis, const struct feature_params& fp, 
 vector1d& fn, vector1d& fn_dr){

    double fn_val, fn_dr_val;
    const double fc = cosine_cutoff_function(dis, fp.cutoff);
    const double fc_dr = cosine_cutoff_function_d(dis, fp.cutoff);

    fn.resize(fp.params.size());
    fn_dr.resize(fp.params.size());
    if (fp.pair_type == "gaussian"){
        for (int n = 0; n < fp.params.size(); ++n){
            gauss_d(dis, fp.params[n][0], fp.params[n][1], fn_val, fn_dr_val);
            fn[n] = fn_val * fc;
            fn_dr[n] = fn_dr_val * fc + fn_val * fc_dr;
        }
    }
}

vector2i get_lm_info(const int& max_l){

    vector2i lm_comb;
    for (int l = 0; l < max_l + 1; ++l){
        for (int m = -l; m < 1; ++m){
            lm_comb.emplace_back(vector1i{l,m,l*l+l+m,l*l+l-m});
        }
    }
    return lm_comb;
}

void get_ylm(const vector1d& sph, const vector2i& lm_info, vector1dc& ylm){

    ylm.resize(lm_info.size());
    for (int lm = 0; lm < lm_info.size(); ++lm){
        ylm[lm] = boost::math::spherical_harmonic
            (lm_info[lm][0], lm_info[lm][1], sph[0], sph[1]);
    }
}

void get_ylm
(const vector1d& sph, const vector2i& lm_info,
 vector1dc& ylm, vector1dc& ylm_dtheta){
// ylm_dphi = i*m*Ylm

    const dc imag(0.0, 1.0);

    const double tan_theta = tanl(sph[0]);
    const double tan_theta_inv = 1.0 / tan_theta;
    const dc exp_imag_phi = exp(-imag*sph[1]);

    ylm.resize(lm_info.size());
    ylm_dtheta.resize(lm_info.size());
    for (int lm = 0; lm < lm_info.size(); ++lm){
        ylm[lm] = boost::math::spherical_harmonic
            (lm_info[lm][0], lm_info[lm][1], sph[0], sph[1]);
    }
    for (int lm = 0; lm < lm_info.size(); ++lm){
        ylm_dtheta[lm] = spherical_harmonic_dtheta
            (lm_info[lm][0], lm_info[lm][1], tan_theta_inv,
             exp_imag_phi, ylm, lm);
    }

}

vector1d cartesian_to_spherical(const vector1d& v){

    bg::model::point<long double,3,bg::cs::cartesian> p1(v[0], v[1], v[2]);
    bg::model::point<long double,3,bg::cs::spherical<bg::radian> > p2;
    bg::transform(p1, p2);
    return vector1d {static_cast<double>(bg::get<1>(p2)),
        static_cast<double>(bg::get<0>(p2))}; // theta, phi
}

dc spherical_harmonic_dtheta
(const int& l, const int& m, const double& tan_theta_inv,
 const dc& exp_imag_phi, const vector1dc& ylm, const int& lm){

    if (m < 0){
        dc ylm_d = sqrt((l-m)*(l+m+1)) * ylm[lm+1] * exp_imag_phi;
        if (std::isinf(tan_theta_inv) == false){
            ylm_d += double(m) * ylm[lm] * tan_theta_inv;
        }
        return ylm_d;
    }
    else if (l > 0 and m == 0){
        return - sqrt(l*(l+1)) * ylm[lm-1] * std::conj(exp_imag_phi);
    }
    else return 0.0;
}
