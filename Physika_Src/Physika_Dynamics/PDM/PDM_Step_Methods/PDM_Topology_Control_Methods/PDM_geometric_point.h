/*
 * @file PDM_geometric_point.h 
 * @brief class PDMGeometricPoint.
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

#ifndef PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_TOPOLOGY_CONTROL_METHODS_PDM_GEOMETRIC_POINT_H
#define PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_TOPOLOGY_CONTROL_METHODS_PDM_GEOMETRIC_POINT_H

#include <set>
#include <vector>

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Utilities/dimension_trait.h"

namespace Physika{

template <typename Scalar, int Dim> class PDMTopologyControlMethod;
class PDMGeometricComponent;
class PDMElementTuple;

template <typename Scalar, int Dim>
class PDMGeometricPoint
{
public:
    PDMGeometricPoint();
    ~PDMGeometricPoint();

    //getter
    unsigned int vertexId() const;
    bool isPotentialSplit() const;
    unsigned int adjoinNum() const;
    const std::set<unsigned int> & adjoinElement();
    const Vector<Scalar, Dim> & lastVertexPos() const;
    unsigned int rigidEleNum() const;

    //setter
    void setVertexId(unsigned int vertex_id);
    void setPotentialSplit(bool potential_split);
    void setLastVertexPos(const Vector<Scalar, Dim> & last_vertex_pos);
    void setRigidEleNum(unsigned int rigid_ele_num);

    void addCrackElementTuple(const PDMElementTuple & ele_tuple);
    void clearCrackElementTuple();

    void addAdjoinElement(unsigned int ele_id);
    void clearAdjoinElement();

    //core function
    bool splitByDisjoinSet(PDMTopologyControlMethod<Scalar, Dim> * topology_control_method);
    void updateGeometricPointPos(PDMTopologyControlMethod<Scalar, Dim> * topology_control_method, Scalar dt);
    bool smoothCrackGeometricPointPos(PDMTopologyControlMethod<Scalar, Dim> * topology_control_method);

protected:
    // utility functions for disjoin set
    bool isElementsConnected(const VolumetricMesh<Scalar, Dim> * mesh, unsigned int fir_ele_id, unsigned int sec_ele_id);

    Vector<Scalar, Dim> calculateAverageVelocity(PDMTopologyControlMethod<Scalar, Dim> * topology_control_method);

protected:
    unsigned int vertex_id_;                                  // vertex id, default: 0

    Vector<Scalar, Dim> last_vertex_pos_;                     // previous vertex position used to impose rigid constraint, default:(0.0, 0.0, 0.0)
    unsigned int rigid_ele_num_;                              // initialized as 0 at every time step, used to impose rigid constrain in topology control method

    std::set<PDMElementTuple> crack_element_tuples_;          // crack element tuples
    std::set<unsigned int> adjoin_elements_;                  // adjoin elements

    bool potential_split_;                                    // if potentially split, default: false, would be true if crack faces added
    bool need_crack_adjust_;                                  // if vertex ever split, then may need adjustment to reduce Zag, default: false
};

} //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_TOPOLOGY_CONTROL_METHODS_PDM_GEOMETRIC_POINT_H
