/*
 * @file PDM_topology_control_method.h 
 * @brief PDMTopologyControlMethod used to control the topology of simulated mesh.
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

#ifndef PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_TOPOLOGY_CONTROL_METHODS_PDM_TOPOLOGY_CONTROL_METHOD_H
#define PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_TOPOLOGY_CONTROL_METHODS_PDM_TOPOLOGY_CONTROL_METHOD_H

#include <vector>
#include <set>
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Topology_Control_Methods/PDM_topology_control_method_base.h"

namespace Physika{

template <typename Scalar, int Dim> class VolumetricMesh;
template <typename Scalar, int Dim> class PDMBase;
template <typename Scalar, int Dim> class PDMGeometricPoint;
template <typename Scalar, int Dim> class Vector;
class PDMElementTuple;

template <typename Scalar, int Dim>
class PDMTopologyControlMethod:public PDMTopologyControlMethodBase<Scalar, Dim>
{
public:
    PDMTopologyControlMethod();
    virtual ~PDMTopologyControlMethod();

    virtual void setMesh(VolumetricMesh<Scalar, Dim> * mesh);     
    virtual void setDriver(PDMBase<Scalar, Dim> * pdm_base);

    void setRotVelDecayRatio(Scalar rot_vel_decay_ratio);
    void setCriticalEleQuality(Scalar critical_ele_quality);
    void setMaxRigidRotDegree(Scalar max_rigid_rot_degree);

    void setCrackSmoothLevel(Scalar crack_smooth_level);
    Scalar crackSmoothLevel() const;

    void enableAdjustMeshVertexPos();
    void disableAdjustMeshVertexPos();
    void enableRotateMeshVertex();
    void disableRotateMeshVertex();
    void enableSmoothCrackVertexPos();
    void disableSmoothCrackVertexPos();
    void enableRigidConstrain();
    void disableRigidConstrain();

    VolumetricMesh<Scalar, Dim> * mesh();
    PDMBase<Scalar, Dim> * driver();

    //core functions
    virtual void addElementTuple(unsigned int fir_ele_id, unsigned int sec_ele_id);           
    void deleteElementTuple(const PDMElementTuple & ele_tuple);
    void addTopologyChange(const std::vector<unsigned int> & topology_change);
    void addGeometricPoint(const PDMGeometricPoint<Scalar, Dim> & gometric_point);

    virtual void topologyControlMethod(Scalar dt);                                           

    const Vector<Scalar, Dim> & rotateVelocity(unsigned int par_id) const;

protected:
    void initGeometricPointsVec();

    void refreshGeometricPointsCrackElementTuple();
    void refreshGeometricPointsAdjoinElement();
    void checkGeometricPointsAdjoinElement();

    void smoothCrackVertexPos();                  //adjust crack vertex position

    void splitGeometricPointsByDisjoinSet();      //split geometric points by DisjoinSet, add topology changing information to topology_changes_
    void changeMeshTopology();                    //modify the topology of tet mesh
    
    void adjustMeshVertexPos();                   //adjust vertex position to particle position
    void refreshParticlesRotVelocity();
    void updateMeshVertexPos(Scalar dt);          //update geometric position of each vertex through physical information  

    void imposeRigidConstrain();                  //impose rigid constrain
    void imposeRigidConstrainByDisjoinSet();      //impose rigid constrain by disjoin set

protected:

    //utility functions for splitting

    // for tet, elements are connected if they share a face
    // for tri, elements are connected if they share a edge
    bool isElementConnected(unsigned int fir_ele_id, unsigned int sec_ele_id, std::vector<unsigned int> & share_vertex_vec);

    //utility functions for rigid constraint
    void initEnvForRigidConstrain();
    Scalar computeElementQuality(Scalar ref_volume, const Vector<Scalar, Dim> & x0, const Vector<Scalar, Dim> & x1, const Vector<Scalar, Dim> & x2, const Vector<Scalar, Dim> & x3) const;

    unsigned int numInvertedElement() const;
    unsigned int numIsolatedElement() const;
    unsigned int numIsolatedAndInvertedElement() const;
    
protected:
    std::vector<PDMGeometricPoint<Scalar, Dim> > gometric_points_;  //note: the index of points in vector is strictly equal to the index in tet mesh
    std::set<PDMElementTuple> crack_element_tuples_;    
    std::vector<std::vector<unsigned int> > topology_changes_;      //each element contain three "unsigned int": ele_id, local_id, new_global_id

    std::vector<Vector<Scalar, Dim> > rot_vel_vec_;   //vector of rotation velocities
    Scalar rot_vel_decay_ratio_;                      //rot_vel *= (1-rot_vel_decay_ratio_), default: 0.0

    Scalar critical_ele_quality_;  //default: 0.3, critical element qualify for imposing rigid constrain
    Scalar max_rigid_rot_degree_;  //default: 10.0, max rotation degree for imposing rigid constrain
    Scalar crack_smooth_level_;    //default: 0.5, [0.0, 1.0], smooth level for crack vertex

    bool enable_adjust_mesh_vertex_pos_;  //default: false
    bool enable_rot_vertex_;              //default: false
    bool enable_smooth_crack_vertex_pos_; //default: false
    bool enable_rigid_constrain_;         //default: false          
};

} // end of namespace Physika

#endif //PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_TOPOLOGY_CONTROL_METHODS_PDM_TOPOLOGY_CONTROL_METHOD_H
