/*
 * @file PDM_bond.h 
 * @Basic PDMBond class. bond based of PDM
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

#ifndef PHYSIKA_DYNAMICS_PDM_PDM_BOND_H
#define PHYSIKA_DYNAMICS_PDM_PDM_BOND_H

#include "Physika_Dynamics/PDM/PDM_base.h"

namespace Physika{

template<typename Scalar, int Dim>
class PDMBond: public PDMBase<Scalar, Dim>
{
public:
	PDMBond();
	PDMBond(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file);
	PDMBond(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file, Scalar bulk_modulus);
	virtual ~PDMBond();

	// getter and setter
	bool isHomogeneousBulkModulus() const;
	Scalar bulkModulus(unsigned int par_idx = 0) const;

	void setBulkModulus(unsigned int par_idx, Scalar bulk_modulus);
	void setHomogeneousBulkModulus(Scalar bulk_modulus);
	void setBulkModulusVec(const std::vector<Scalar>& bulk_modulus_vec);

protected:
	bool is_homogeneous_bulk_modulus_;       // default: true
	std::vector<Scalar> bulk_modulus_vec_;   // bulk modulus parameter vector, it contain only one member in the 
	                                         // homogeneous case, while the size of vector must be no less
	                                         // than vertex number for inhomogeneous material.
};

}// end of namespace Physika

#endif //PHYSIKA_DYNAMICS_PDM_PDM_BOND_H