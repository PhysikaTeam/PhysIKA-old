/*
 * @file fem_solid_force_model.h 
 * @Brief the "engine" for fem solid drivers.
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

#ifndef PHYSIKA_DYNAMICS_FEM_FEM_SOLID_FORCE_MODEL_FEM_SOLID_FORCE_MODEL_H_
#define PHYSIKA_DYNAMICS_FEM_FEM_SOLID_FORCE_MODEL_FEM_SOLID_FORCE_MODEL_H_

#include <vector>

namespace Physika{

/*
 * the "engine" for fem solid drivers, do the actual force computation 
 * the drivers pass data of simulation to the force model, and the force
 * model return force, force differential, etc. to the drivers
 *
 * Note: with this design we're able to integrate fem computation for different element types 
 *         into one driver class without messing up the code
 */

template <typename Scalar, int Dim> class VolumetricMesh;
template <typename Scalar, int Dim> class ConstitutiveModel;
template <typename Scalar, int Dim> class Vector;

template <typename Scalar, int Dim>
class FEMSolidForceModel
{
public:
    //construct force model with volumetric mesh and constitutive model, the constitutive model can be uniform, region-wise, and element-wise on the mesh
    FEMSolidForceModel(const VolumetricMesh<Scalar,Dim> &simulation_mesh, const std::vector<ConstitutiveModel<Scalar,Dim>*> &constitutive_model);
    virtual ~FEMSolidForceModel();

    //given world space coordinates of mesh vertices, compute the internal forces on the entire mesh
    virtual void computeGlobalInternalForces(const std::vector<Vector<Scalar,Dim> > &current_vert_pos, std::vector<Vector<Scalar,Dim> > &force) const = 0;
    //compute internal forces on vertices of a specific element
    virtual void computeElementInternalForces(unsigned int ele_idx, const std::vector<Vector<Scalar,Dim> > &current_vert_pos, std::vector<Vector<Scalar,Dim> > &force) const = 0;
    //compute force differentials, for implicit time stepping
    virtual void computeGlobalInternalForceDifferentials(const std::vector<Vector<Scalar,Dim> > &current_vert_pos, 
                                                                                    const std::vector<Vector<Scalar,Dim> > &vert_pos_differentials,
                                                                                     std::vector<Vector<Scalar,Dim> > &force_differentials) const = 0;
    virtual void computeElementInternalForceDifferentials(unsigned int ele_idx, const std::vector<Vector<Scalar,Dim> > &current_vert_pos,
                                                                                       const std::vector<Vector<Scalar,Dim> > &vert_pos_differentials,
                                                                                       std::vector<Vector<Scalar,Dim> > &force_differentials) const = 0;
protected:
    const ConstitutiveModel<Scalar,Dim>& elementMaterial(unsigned int ele_idx) const;
protected:
    FEMSolidForceModel(); //prohibit default constructor
    const VolumetricMesh<Scalar,Dim> &simulation_mesh_;
    const std::vector<ConstitutiveModel<Scalar,Dim>*> &constitutive_model_;
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_FEM_FEM_SOLID_FORCE_MODEL_FEM_SOLID_FORCE_MODEL_H_
