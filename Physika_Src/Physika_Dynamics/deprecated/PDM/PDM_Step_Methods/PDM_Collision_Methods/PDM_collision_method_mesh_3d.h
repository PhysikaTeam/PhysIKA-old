/*
 * @file PDM_collision_method_mesh_3d.h 
 * @brief class of collision method(three dim) based on mesh for PDM drivers.
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

#ifndef PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_COLLISION_METHODS_PDM_COLLISION_METHOD_MESH_3D_H
#define PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_COLLISION_METHODS_PDM_COLLISION_METHOD_MESH_3D_H

#include <vector>
#include <set>

#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Collision_Methods/PDM_collision_method_mesh.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Collision_Methods/PDM_collision_method_grid_3d.h"

namespace Physika{

template <typename Scalar, int Dim> class Vector;
template <typename Scalar, int Dim> class SquareMatrix;

template <typename Scalar>
class PDMCollisionMethodMesh<Scalar, 3>:public PDMCollisionMethodGrid<Scalar,3>
{
public:
    PDMCollisionMethodMesh();
    ~PDMCollisionMethodMesh();

    virtual void setDriver(PDMBase<Scalar,3> * driver);

protected:
    virtual void locateParticleBin();
    virtual void collisionDectectionAndResponse();

protected:

    void refreshPreStoredMeshInfomation();
    bool intersectTetrahedra(unsigned int fir_ele_id, unsigned int sec_ele_id) const;

protected:
    //pre stored mesh information to improve efficiency 
    std::vector<std::vector<unsigned int> > mesh_ele_index_vec_;
    std::vector<std::vector<Vector<Scalar, 3> > > mesh_ele_pos_vec_;

};

inline bool operator < (const std::pair<unsigned int, unsigned int> & lhs, const std::pair<unsigned int, unsigned int> & rhs)
{
    if (lhs.first < rhs.first) return true;
    if (lhs.first > rhs.first) return false;
    if (lhs.second < rhs.second) return true;
    if (lhs.second > rhs.second) return false;
}

}// end of namespace Phyaika
#endif //PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_COLLISION_METHODS_PDM_COLLISION_METHOD_MESH_3D_H