/*
 * @file quad_cubic_mesh_fem_solid_force_model.h 
 * @Brief fem solid force model for quad and cubic mesh.
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_DYNAMICS_FEM_FEM_SOLID_FORCE_MODEL_QUAD_CUBIC_MESH_FEM_SOLID_FORCE_MODEL_H_
#define PHYSIKA_DYNAMICS_FEM_FEM_SOLID_FORCE_MODEL_QUAD_CUBIC_MESH_FEM_SOLID_FORCE_MODEL_H_

#include "Physika_Dynamics/FEM/FEM_Solid_Force_Model/fem_solid_force_model.h"

namespace Physika{

template <typename Scalar, int Dim>
class QuadCubicMeshFEMSolidForceModel: public FEMSolidForceModel<Scalar,Dim>
{
public:
    QuadCubicMeshFEMSolidForceModel();
    explicit QuadCubicMeshFEMSolidForceModel(const FEMSolid<Scalar,Dim> *fem_solid_driver);
    ~QuadCubicMeshFEMSolidForceModel();
protected:
};
    
} //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_FEM_FEM_SOLID_FORCE_MODEL_QUAD_CUBIC_MESH_FEM_SOLID_FORCE_MODEL_H_
