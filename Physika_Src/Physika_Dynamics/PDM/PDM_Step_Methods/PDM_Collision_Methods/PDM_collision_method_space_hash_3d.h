/*
 * @file PDM_collision_method_space_hash_3d.h 
 * @brief class of collision method(two dim) for PDM drivers.
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

#ifndef PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_COLLISION_METHODS_PDM_COLLISION_METHOD_SPACE_HASH_3D_H
#define PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_COLLISION_METHODS_PDM_COLLISION_METHOD_SPACE_HASH_3D_H

#include <vector>
#include <set>
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Collision_Methods/PDM_collision_method_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Collision_Methods/PDM_collision_method_space_hash.h"

namespace Physika{

template <typename Scalar, int Dim> class VolumetricMesh;
template <typename Scalar, int Dim> class Vector;

template<typename Scalar>
class PDMCollisionMethodSpaceHash<Scalar, 3>:public PDMCollisionMethodBase<Scalar, 3>
{
public:
    PDMCollisionMethodSpaceHash();
    ~PDMCollisionMethodSpaceHash();

    Scalar gridCellSize() const;
    unsigned int hashTableSize() const;

    void enableEdgeIntersect();
    void disableEdgeIntersect();
    void enableOverlapVolCollisionResponse();
    void disableOverlapVolCollisionResponse();
    virtual void collisionMethod();

    void setHashTableSize(unsigned int hash_table_size);
    void setGridCellSize(Scalar grid_cell_size);
    void autoSetGridCellSize();

protected:
    void resetHashTable();
    void locateVertexToHashTable();
    void locateElementToHashTable();

    //need further consideration
    virtual void collisionDetectionAndResponseViaVertexPenetrate();
    virtual void collisionDetectionAndResponseViaEdgeIntersect();

protected:

    virtual void collisionResponse(const std::set<std::pair<unsigned int, unsigned int> > & collision_result) ;

    unsigned int hashFunction(long long i, long long j, long long k) const;
    void refreshPreStoredMeshInfomation();

    //utility function
    void generateVertexAdjoinElement(const VolumetricMesh<Scalar, 3> * mesh, std::vector<std::vector<unsigned int> > & vert_adjoin_ele_vec);
    bool isVertexPenetrateElement(const SquareMatrix<Scalar, 3> & A_inverse, const Vector<Scalar, 3> & x0, const Vector<Scalar, 3> & p) const;
    bool isElementContainVertex(unsigned int ele_id, unsigned int vert_id) const;
    bool isVertexInsideBoundingVolume(unsigned int ele_id, const Vector<Scalar, 3> & vert_pos) const;

    bool intersectTethedraViaVertexPenetrate(unsigned int ele_id, unsigned int vert_id, const SquareMatrix<Scalar, 3> & A_inverse, const Vector<Scalar, 3> & x0) const;

    void generateElementFaceNormal(const VolumetricMesh<Scalar, 3> * mesh, std::vector<std::vector<Vector<Scalar, 3> > > & ele_face_normal_vec);
    bool isBoundingVolumeOverlap(unsigned int fir_ele_id, unsigned int sec_ele_id);

    bool intersectTetrahedraViaEdgeIntersect(unsigned int fir_ele_id, unsigned int sec_ele_id, 
                                             const std::vector<Vector<Scalar, 3> > & fir_ele_face_normal, const std::vector<Vector<Scalar, 3> > & sec_ele_face_normal,
                                             std::vector<unsigned char> & masks,
                                             std::vector<std::vector<Scalar> > & coord,
                                             std::vector<Vector<Scalar,3> > & teta_to_tetb_vec,
                                             std::vector<Vector<Scalar,3> > & tetb_to_teta_vec);

    Scalar tetrahedraOverlapVolume(unsigned int fir_ele_id, unsigned int sec_ele_id);

protected:
    std::vector<std::vector<unsigned int> > vertex_hash_table_;
    std::vector<std::vector<unsigned int> > element_hash_table_;
    Scalar grid_cell_size_;   //default: 0.1

    //pre stored mesh information to improve efficiency 
    std::vector<std::vector<unsigned int> > mesh_ele_index_vec_;
    std::vector<std::vector<Vector<Scalar, 3> > > mesh_ele_pos_vec_;

    //bounding volume information to improve efficiency
    std::vector<Vector<Scalar, 3> > mesh_ele_min_corner_vec_;
    std::vector<Vector<Scalar, 3> > mesh_ele_max_corner_vec_;

    bool use_edge_intersect_;                 //default: false, exactly test whether two tet insecsect 
    bool use_overlap_vol_collision_response_; //default: true
};

}// end of namespace Physika

#endif //PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_COLLISION_METHODS_PDM_COLLISION_METHOD_SPACE_HASH_3D_H