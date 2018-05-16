/*
 * @file PDM_bond_2d.h 
 * @Basic PDMBond class(two dimension). bond based of PDM
 * @author Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_DYNAMICS_PDM_PDM_BOND_2D_H
#define PHYSIKA_DYNAMICS_PDM_PDM_BOND_2D_H

#include "Physika_Dynamics/PDM/PDM_base.h"
#include "Physika_Dynamics/PDM/PDM_bond.h"

namespace Physika{

// class template specializations
template <typename Scalar>
class PDMBond<Scalar, 2>: public PDMBase<Scalar, 2>
{
public:
    PDMBond();
    PDMBond(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file);
    PDMBond(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file, Scalar bulk_modulus, Scalar thickness);
    virtual ~PDMBond();

    // getter and setter
    bool isHomogeneousBulkModulus() const;
    Scalar bulkModulus(unsigned int par_idx = 0) const;
    Scalar thickness() const;

    void setBulkModulus(unsigned int par_idx, Scalar bulk_modulus);
    void setHomogeneousBulkModulus(Scalar bulk_modulus);
    void setBulkModulusVec(const std::vector<Scalar>& bulk_modulus_vec);
    void setThickness(Scalar thickness);

protected:
    bool is_homogeneous_bulk_modulus_;       // default: true
    std::vector<Scalar> bulk_modulus_vec_;   // bulk modulus parameter vector, it contain only one member in the 
                                             // homogeneous case, while the size of vector must be no less
                                             // than vertex number for inhomogeneous material.
    Scalar thickness_;  //default: 1.0
};

}// end of namespace Physika

#endif //PHYSIKA_DYNAMICS_PDM_PDM_BOND_2D_H