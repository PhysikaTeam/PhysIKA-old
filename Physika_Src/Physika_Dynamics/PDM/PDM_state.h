/*
 * @file PDM_state.h 
 * @Basic PDMState class. state based of PDM
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

#ifndef PHYSIKA_DYNAMICS_PDM_PDM_STATE_H
#define PHYSIKA_DYNAMICS_PDM_PDM_STATE_H

#include "Physika_Dynamics/PDM/PDM_bond.h"

namespace Physika{

template<typename Scalar, int Dim>
class PDMState: public PDMBond<Scalar, Dim>
{
public:
	PDMState();
	PDMState(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file);
	PDMState(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file, Scalar bulk_modulus, Scalar shear_modulus);
	virtual ~PDMState();

	// getter and setter
	bool isHomogeneousShearModulus() const;
	Scalar shearModulus(unsigned int par_idx = 0) const;

	void setShearModulus(unsigned int par_idx, Scalar mu);
	void setHomogeneousShearModulus(Scalar mu);
	void setShearModulusVec(const std::vector<Scalar> mu_vec);

protected:
	bool is_homogeneous_shear_modulus_;
	std::vector<Scalar> shear_modulus_vec_; // mu (shear modulus) parameter vector, similar to bulk_modulus_vec_
};

}// end of namespace Physika

#endif // PHYSIKA_DYNAMICS_PDM_PDM_STATE_H