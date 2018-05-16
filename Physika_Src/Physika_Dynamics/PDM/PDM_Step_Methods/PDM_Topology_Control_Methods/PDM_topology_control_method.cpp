/*
 * @file PDM_topology_control_method_base.cpp 
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

#include <iostream>
#include "Physika_Dependency/Eigen/Eigen"

#include "Physika_Core/Timer/timer.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Geometry/Volumetric_Meshes/tet_mesh.h"
#include "Physika_Dynamics/PDM/PDM_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Topology_Control_methods/PDM_topology_control_method.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Topology_Control_methods/PDM_geometric_point.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Topology_Control_methods/PDM_element_tuple.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Topology_Control_Methods/PDM_disjoin_set.h"

namespace Physika{

template <typename Scalar, int Dim>
PDMTopologyControlMethod<Scalar, Dim>::PDMTopologyControlMethod()
    :rot_vel_decay_ratio_(0.0), critical_ele_quality_(0.3),
    max_rigid_rot_degree_(10.0), crack_smooth_level_(0.5),
    enable_adjust_mesh_vertex_pos_(false), enable_rot_vertex_(false), 
    enable_smooth_crack_vertex_pos_(false), enable_rigid_constrain_(false)
{

}

template <typename Scalar, int Dim>
PDMTopologyControlMethod<Scalar, Dim>::~PDMTopologyControlMethod()
{

}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::setMesh(VolumetricMesh<Scalar, Dim> * mesh)
{
    this->mesh_ = mesh;
    PHYSIKA_ASSERT(this->mesh_);
    this->initGeometricPointsVec();
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::setDriver(PDMBase<Scalar, Dim> * pdm_base)
{
    this->pdm_base_ = pdm_base;
    PHYSIKA_ASSERT(this->pdm_base_);
    this->mesh_ = this->pdm_base_->mesh();
    PHYSIKA_ASSERT(this->mesh_);
    this->initGeometricPointsVec();

    //initialize rot_vel_vec_ for particles
    this->rot_vel_vec_.resize(this->pdm_base_->numSimParticles());
    for (unsigned int par_id = 0; par_id < this->pdm_base_->numSimParticles(); par_id++)
        this->rot_vel_vec_[par_id] = Vector<Scalar, Dim>(0.0);
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::setRotVelDecayRatio(Scalar rot_vel_decay_ratio)
{
    this->rot_vel_decay_ratio_ = rot_vel_decay_ratio;
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::setCriticalEleQuality(Scalar critical_ele_quality)
{
    this->critical_ele_quality_ = critical_ele_quality;
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::setMaxRigidRotDegree(Scalar max_rigid_rot_degree)
{
    this->max_rigid_rot_degree_ = max_rigid_rot_degree;
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::setCrackSmoothLevel(Scalar crack_smooth_level)
{
    PHYSIKA_ASSERT(crack_smooth_level >= 0.0 && crack_smooth_level <= 1.0);
    this->crack_smooth_level_ = crack_smooth_level;
}

template <typename Scalar, int Dim>
Scalar PDMTopologyControlMethod<Scalar, Dim>::crackSmoothLevel() const
{
    return this->crack_smooth_level_;
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::enableAdjustMeshVertexPos()
{
    this->enable_adjust_mesh_vertex_pos_ = true;
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::disableAdjustMeshVertexPos()
{
    this->enable_adjust_mesh_vertex_pos_ = false;
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::enableRotateMeshVertex()
{
    this->enable_rot_vertex_ = true;
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::disableRotateMeshVertex()
{
    this->enable_rot_vertex_ = false;
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::enableSmoothCrackVertexPos()
{
    this->enable_smooth_crack_vertex_pos_ = true;
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::disableSmoothCrackVertexPos()
{
    this->enable_smooth_crack_vertex_pos_ = false;
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::enableRigidConstrain()
{
    this->enable_rigid_constrain_ = true;
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::disableRigidConstrain()
{
    this->enable_rigid_constrain_ = false;
}

template <typename Scalar, int Dim>
VolumetricMesh<Scalar, Dim> * PDMTopologyControlMethod<Scalar, Dim>::mesh()
{
    return this->mesh_;
}

template <typename Scalar, int Dim>
PDMBase<Scalar, Dim> * PDMTopologyControlMethod<Scalar, Dim>::driver()
{
    return this->pdm_base_;
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::addTopologyChange(const std::vector<unsigned int> & topology_change)
{
    PHYSIKA_ASSERT(topology_change.size() == 3);
    this->topology_changes_.push_back(topology_change);
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::addElementTuple(unsigned int fir_ele_id, unsigned int sec_ele_id)
{
    std::vector<unsigned int>  share_vertex_vec;
    if (this->isElementConnected(fir_ele_id, sec_ele_id, share_vertex_vec) == true)
    {
        PHYSIKA_ASSERT(share_vertex_vec.size()==2 || share_vertex_vec.size()==3);
        PDMElementTuple ele_tuple;
        ele_tuple.setElementVec(fir_ele_id, sec_ele_id);
        this->crack_element_tuples_.insert(ele_tuple);
    }
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::deleteElementTuple(const PDMElementTuple & ele_tuple)
{
    this->crack_element_tuples_.erase(ele_tuple);
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::addGeometricPoint(const PDMGeometricPoint<Scalar, Dim> & gometric_point)
{
    PHYSIKA_ASSERT(gometric_point.vertexId() == this->gometric_points_.size());
    this->gometric_points_.push_back(gometric_point);
}

template <typename Scalar, int Dim>
const Vector<Scalar, Dim> & PDMTopologyControlMethod<Scalar, Dim>::rotateVelocity(unsigned int par_id) const
{
    return this->rot_vel_vec_[par_id];
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::topologyControlMethod(Scalar dt)
{
    Timer timer;
    Timer timer2;
    timer.startTimer();

    timer2.startTimer();

    //split, change topology, and refresh adjoin elements
    refreshGeometricPointsCrackElementTuple();
    splitGeometricPointsByDisjoinSet();
    changeMeshTopology();
    //refreshGeometricPointsAdjoinElement();
    //checkGeometricPointsAdjoinElement();

    timer2.stopTimer();
    std::cout<<"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n";
    std::cout<<"time for split: "<<timer2.getElapsedTime()<<std::endl;


    if (this->enable_rigid_constrain_)
        initEnvForRigidConstrain();
      

    timer2.startTimer();
    //smooth crack vertex pos
    if (this->enable_smooth_crack_vertex_pos_)
        smoothCrackVertexPos();
    timer2.stopTimer();
    std::cout<<"time for smooth crack vertex: "<<timer2.getElapsedTime()<<std::endl;

    //adjust the vertex pos to particle position
    if (this->enable_adjust_mesh_vertex_pos_)
        adjustMeshVertexPos();

    //refresh rotation velocity
    if (this->enable_rot_vertex_)
        refreshParticlesRotVelocity();


    timer2.startTimer();
    //update mesh vertex position
    updateMeshVertexPos(dt);
    timer2.stopTimer();
    std::cout<<"time for update vertex: "<<timer2.getElapsedTime()<<std::endl;
    std::cout<<"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n";

    //further adjust mesh vertex to impose rigid constrain if necessary
    if (this->enable_rigid_constrain_)
    {
        //imposeRigidConstrain()
        imposeRigidConstrainByDisjoinSet();
    }

    std::cout<<"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n";
    std::cout<<"isolated num: "<<this->numIsolatedElement()<<std::endl;
    std::cout<<"inverted num: "<<this->numInvertedElement()<<std::endl;
    std::cout<<"isolated & inverted num: "<<this->numIsolatedAndInvertedElement()<<std::endl;
    std::cout<<"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n";

    timer.stopTimer();
    std::cout<<"--------------------------------------------------------------------"<<std::endl;
    std::cout<<"time cost for topology control method: "<<timer.getElapsedTime()<<std::endl;
    std::cout<<"--------------------------------------------------------------------"<<std::endl;
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::initGeometricPointsVec()
{
    unsigned int vertex_num = this->mesh_->vertNum();
    this->gometric_points_.clear();
    this->gometric_points_.reserve(vertex_num);
    for (unsigned int ver_id = 0; ver_id<vertex_num; ver_id++)
    {
        PDMGeometricPoint<Scalar, Dim> gometric_point;
        gometric_point.setVertexId(ver_id);               
        //note: the index of points in vector is strictly equal to the index in tet mesh
        this->gometric_points_.push_back(gometric_point);
    }

    //initially refresh adjoin elements of points
    this->refreshGeometricPointsAdjoinElement();
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::refreshGeometricPointsAdjoinElement()
{
    //clear adjoin elements
    for (unsigned int i=0; i<this->gometric_points_.size(); i++)
        this->gometric_points_[i].clearAdjoinElement();
    
    //add adjoin elements to geometric point
    unsigned int ele_num = this->mesh_->eleNum();
    for (unsigned int ele_id=0; ele_id<ele_num; ele_id++)
    {
        std::vector<unsigned int> ele_vert;
        this->mesh_->eleVertIndex(ele_id, ele_vert);

        PHYSIKA_ASSERT(ele_vert.size() == 3 || ele_vert.size() == 4);
        for (unsigned int vert_id = 0; vert_id<ele_vert.size(); vert_id++)
            this->gometric_points_[ele_vert[vert_id]].addAdjoinElement(ele_id);
    }
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::checkGeometricPointsAdjoinElement()
{
    std::vector<std::set<unsigned int> > gometric_points_adjoin_element(this->gometric_points_.size());

    unsigned int ele_num = this->mesh_->eleNum();
    for (unsigned int ele_id=0; ele_id<ele_num; ele_id++)
    {
        std::vector<unsigned int> ele_vert;
        this->mesh_->eleVertIndex(ele_id, ele_vert);

        PHYSIKA_ASSERT(ele_vert.size() == 3 || ele_vert.size() == 4);
        for (unsigned int vert_id = 0; vert_id<ele_vert.size(); vert_id++)
            gometric_points_adjoin_element[ele_vert[vert_id]].insert(ele_id);
    }

    for (unsigned int i = 0; i < this->gometric_points_.size(); i++)
    {
        if (gometric_points_adjoin_element[i] != this->gometric_points_[i].adjoinElement())
        {
            std::cerr<<"error: geometric points check adjoin elements faild!\n";
            std::exit(EXIT_FAILURE);
        }
    }

}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::refreshGeometricPointsCrackElementTuple()
{
    for(unsigned int i = 0; i < this->gometric_points_.size(); i++)
        this->gometric_points_[i].clearCrackElementTuple();

    for (std::set<PDMElementTuple>::const_iterator iter = this->crack_element_tuples_.begin(); iter != this->crack_element_tuples_.end(); iter++)
    {
        const std::vector<unsigned int> & ele_vec = iter->eleVec();
        std::vector<unsigned int>  share_vertex_vec;
        if (this->isElementConnected(ele_vec[0], ele_vec[1], share_vertex_vec) == true)
        {
            PHYSIKA_ASSERT(share_vertex_vec.size()==2 || share_vertex_vec.size()==3);
            for (unsigned int i=0; i<share_vertex_vec.size(); i++)
                this->gometric_points_[share_vertex_vec[i]].addCrackElementTuple(*iter);
        }
        else
        {
            std::cerr<<"error: element tuple is not immediate neighbors, there exist a bug!\n";
            std::exit(EXIT_FAILURE);
        }
    }
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::refreshParticlesRotVelocity()
{
    if (Dim != 3)
    {
        std::cerr<<"error: function currently is not implemented for two dimension!\n";
        std::exit(EXIT_FAILURE);
    }

    Scalar max_rot_vel = 0.0;
    unsigned int max_par_id = 0;
    
    for (unsigned int par_id = 0; par_id < this->rot_vel_vec_.size(); par_id++)
    {
        //decay rotation velocity
        this->rot_vel_vec_[par_id] *= (1.0 - this->rot_vel_decay_ratio_);

        const PDMParticle<Scalar, Dim> & particle = this->pdm_base_->particle(par_id);

        unsigned int adjoin_num = particle.validFamilySize();
        if (adjoin_num < 1) continue;

        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> A(3*adjoin_num, 3);
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> b(3*adjoin_num, 1);

        A.setZero();
        b.setZero();

        const Vector<Scalar, Dim> & particle_pos = this->pdm_base_->particleCurrentPosition(par_id);
        const Vector<Scalar, Dim> & particle_trans_vel = this->pdm_base_->particleVelocity(par_id);

        unsigned int j_idx = 0; 
        std::list<PDMFamily<Scalar,Dim> > & family = this->pdm_base_->particle(par_id).family();
        for (std::list<PDMFamily<Scalar,Dim> >::iterator j_iter = family.begin(); j_iter != family.end(); j_iter++)
        {
            // skip invalid family member
            if (j_iter->isVaild() == false || j_iter->isCrack() == true ) continue;

            const Vector<Scalar, Dim> & j_pos = this->pdm_base_->particleCurrentPosition(j_iter->id()) - particle_pos;
            A(3*j_idx, 1)   =  j_pos[2];
            A(3*j_idx, 2)   = -j_pos[1];
            A(3*j_idx+1, 0) = -j_pos[2];
            A(3*j_idx+1, 2) =  j_pos[0];
            A(3*j_idx+2, 0) =  j_pos[1];
            A(3*j_idx+2, 1) = -j_pos[0];

            const Vector<Scalar, Dim> & j_vel = this->pdm_base_->particleVelocity(j_iter->id());
            b(3*j_idx)   = j_vel[0] - particle_trans_vel[0];
            b(3*j_idx+1) = j_vel[1] - particle_trans_vel[1];
            b(3*j_idx+2) = j_vel[2] - particle_trans_vel[2];

            j_idx++;
        }

        if (j_idx != adjoin_num)
        {
            std::cerr<<"error: the valid size is incorrect!\n";
            std::exit(EXIT_FAILURE);
        }

        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> rot_vel(3, 1);
        rot_vel = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

        // rotation velocity
        this->rot_vel_vec_[par_id][0] = rot_vel(0);
        this->rot_vel_vec_[par_id][1] = rot_vel(1);
        this->rot_vel_vec_[par_id][2] = rot_vel(2);
        

        if (max_rot_vel < this->rot_vel_vec_[par_id].norm())
        {
            max_rot_vel = this->rot_vel_vec_[par_id].norm();
            max_par_id = par_id;
        }
    }

    std::cout<<"+++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    std::cout<<"max_rot_vel: "<<max_rot_vel<<std::endl;
    std::cout<<"max_par_id: "<<max_par_id<<std::endl;
    std::cout<<"+++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::smoothCrackVertexPos()
{
    unsigned int num_crack_smooth = 0;
    unsigned int vertex_num = this->gometric_points_.size();
    for (unsigned int vert_id = 0; vert_id<vertex_num; vert_id++)
    {
        if (this->gometric_points_[vert_id].smoothCrackGeometricPointPos(this))
            num_crack_smooth++;
    }
    std::cout<<"=============================================================\n";
    std::cout<<"num_crack_smooth: "<<num_crack_smooth<<std::endl;
    std::cout<<"=============================================================\n";
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::splitGeometricPointsByDisjoinSet()
{
    unsigned int num_split = 0;
    unsigned int num_geometric_point = this->gometric_points_.size();
    for (unsigned int ver_id = 0; ver_id<num_geometric_point; ver_id++)
    {
        if (this->gometric_points_[ver_id].splitByDisjoinSet(this) == true)
            num_split++;
    }
    std::cout<<"=============================================================\n";
    std::cout<<"num_split: "<<num_split<<std::endl;
    std::cout<<"=============================================================\n";
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::changeMeshTopology()
{
    //topology modification
    for (unsigned int i=0; i<this->topology_changes_.size(); i++)
    {
        PHYSIKA_ASSERT(topology_changes_[i].size() == 3);
        std::vector<unsigned int> ele_vert;
        this->mesh_->eleVertIndex(topology_changes_[i][0], ele_vert);
        bool contain_vertex = false;
        for (unsigned int j=0; j<ele_vert.size(); j++)
        {
            if (ele_vert[j] == topology_changes_[i][1])
            {
                contain_vertex = true;
                this->mesh_->setEleVertIndex(topology_changes_[i][0], j, topology_changes_[i][2]);
                break;
            }
        }
        if (contain_vertex == false)
        {
            std::cerr<<"error: can't find the vertex in element!\n";
            std::exit(EXIT_FAILURE);
        }
        
    }

    //clear
    this->topology_changes_.clear();
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::adjustMeshVertexPos()
{
    unsigned int vertex_num = this->gometric_points_.size();
    std::vector<Vector<Scalar, Dim> > vertex_weighted_displacement_vec(vertex_num, Vector<Scalar, Dim>(0.0));
    std::vector<Scalar> vertex_weight_vec(vertex_num, 0.0);

    PHYSIKA_ASSERT(this->pdm_base_->numSimParticles() == this->mesh_->eleNum());
    for(unsigned int ele_id = 0; ele_id<this->pdm_base_->numSimParticles(); ele_id++)
    {
        const PDMParticle<Scalar, Dim> & particle = this->pdm_base_->particle(ele_id);
        Scalar weight = particle.mass();
        unsigned int valid_num = particle.validFamilySize();

        //only adjust isolate tet
        //if (valid_num > 0) continue;

        const Vector<Scalar, Dim> & particle_pos = this->pdm_base_->particleCurrentPosition(ele_id);

        std::vector<Vector<Scalar, Dim> > ele_pos_vec;
        this->mesh_->eleVertPos(ele_id, ele_pos_vec);
        PHYSIKA_ASSERT(ele_pos_vec.size() == 3 || ele_pos_vec.size() == 4);

        Vector<Scalar, Dim> centeral_pos(0.0);
        for (unsigned int i=0; i<ele_pos_vec.size(); i++) centeral_pos += ele_pos_vec[i];
        centeral_pos /= ele_pos_vec.size();

        Vector<Scalar, Dim> weighted_displacement = weight*(particle_pos - centeral_pos);

        std::vector<unsigned int> ele_vert;
        this->mesh_->eleVertIndex(ele_id, ele_vert);
        PHYSIKA_ASSERT(ele_vert.size() == 3 || ele_vert.size() == 4);

        for (unsigned int i=0; i<ele_vert.size(); i++) 
        {
            vertex_weighted_displacement_vec[ele_vert[i]] += weighted_displacement;
            vertex_weight_vec[ele_vert[i]] += weight;
        }
    }

    for (unsigned int vert_id = 0; vert_id<vertex_num; vert_id++)
    {
        Vector<Scalar, Dim> new_vert_pos = this->mesh_->vertPos(vert_id) + vertex_weighted_displacement_vec[vert_id]/vertex_weight_vec[vert_id];
        this->mesh_->setVertPos(vert_id, new_vert_pos);
    }

}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::updateMeshVertexPos(Scalar dt)
{
    #pragma omp parallel for
    for (long long ver_id = 0; ver_id<this->gometric_points_.size(); ver_id++)
        this->gometric_points_[ver_id].updateGeometricPointPos(this, dt);
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::imposeRigidConstrain()
{
    //only 3D-case is supported.
    PHYSIKA_ASSERT(Dim == 3);

    unsigned int rigid_ele_num = 0;

    for (unsigned int ele_id = 0; ele_id<this->mesh_->eleNum(); ele_id++)
    {
        Scalar ref_volume = this->pdm_base_->particle(ele_id).volume();

        std::vector<Vector<Scalar, Dim> > ele_vert_pos;
        std::vector<unsigned int> ele_vert_index;

        this->mesh_->eleVertPos(ele_id, ele_vert_pos);
        this->mesh_->eleVertIndex(ele_id, ele_vert_index);

        PHYSIKA_ASSERT(ele_vert_pos.size() == 4);
        PHYSIKA_ASSERT(ele_vert_index.size() == 4);

        Scalar ele_quality = this->computeElementQuality(ref_volume, ele_vert_pos[0], ele_vert_pos[1], ele_vert_pos[2], ele_vert_pos[3]);

        unsigned int x0_index = ele_vert_index[0];
        unsigned int x1_index = ele_vert_index[1];
        unsigned int x2_index = ele_vert_index[2];
        unsigned int x3_index = ele_vert_index[3];

        const Vector<Scalar, Dim> & m_X0_pos = this->gometric_points_[x0_index].lastVertexPos();
        const Vector<Scalar, Dim> & m_X1_pos = this->gometric_points_[x1_index].lastVertexPos();
        const Vector<Scalar, Dim> & m_X2_pos = this->gometric_points_[x2_index].lastVertexPos();
        const Vector<Scalar, Dim> & m_X3_pos = this->gometric_points_[x3_index].lastVertexPos();

        Scalar m_ele_quality = this->computeElementQuality(ref_volume, m_X0_pos, m_X1_pos, m_X2_pos, m_X3_pos);

        //need further consideration
        if (ele_quality < 0.3 && ele_quality < m_ele_quality) 
        {
            rigid_ele_num++;

            Vector<Scalar, Dim> xc(0.0);
            for (unsigned int i=0; i<ele_vert_pos.size(); i++) xc += ele_vert_pos[i];
            xc /= ele_vert_pos.size();

            Vector<Scalar, Dim> x0_minus_xc = ele_vert_pos[0] - xc;
            Vector<Scalar, Dim> x1_minus_xc = ele_vert_pos[1] - xc;
            Vector<Scalar, Dim> x2_minus_xc = ele_vert_pos[2] - xc;

            Eigen::Matrix<Scalar, Dim, Dim> Ds;
            for (unsigned int i = 0; i<Dim; i++) Ds(i, 0) = x0_minus_xc[i];
            for (unsigned int i = 0; i<Dim; i++) Ds(i, 1) = x1_minus_xc[i];
            for (unsigned int i = 0; i<Dim; i++) Ds(i, 2) = x2_minus_xc[i];

            Vector<Scalar, Dim> m_Xc = (m_X0_pos + m_X1_pos + m_X2_pos + m_X3_pos)/4;

            Vector<Scalar, Dim> m_X0_minus_Xc = m_X0_pos - m_Xc;
            Vector<Scalar, Dim> m_X1_minus_Xc = m_X1_pos - m_Xc;
            Vector<Scalar, Dim> m_X2_minus_Xc = m_X2_pos - m_Xc;
            Vector<Scalar, Dim> m_X3_minus_Xc = m_X3_pos - m_Xc;

            Eigen::Matrix<Scalar, Dim, Dim> Dm;
            for (unsigned int i = 0; i<Dim; i++) Dm(i, 0) = m_X0_minus_Xc[i];
            for (unsigned int i = 0; i<Dim; i++) Dm(i, 1) = m_X1_minus_Xc[i];
            for (unsigned int i = 0; i<Dim; i++) Dm(i, 2) = m_X2_minus_Xc[i];

            Eigen::Matrix<Scalar, Dim, Dim> F = Ds*Dm.inverse();
            Eigen::JacobiSVD<Eigen::Matrix<Scalar, Dim, Dim> > svd(F, Eigen::ComputeThinU|Eigen::ComputeThinV);

            Eigen::Matrix<Scalar, Dim, Dim> U = svd.matrixU();
            Eigen::Matrix<Scalar, Dim, Dim> V = svd.matrixV();
            Eigen::Matrix<Scalar, Dim, Dim> R = U*V.transpose();

            SquareMatrix<Scalar, Dim> Rot;
            for (unsigned int i=0; i<3; i++)
            for (unsigned int j=0; j<3; j++)
                Rot(i, j) = R(i, j);

            Vector<Scalar, Dim> new_x0_pos = xc + Rot*m_X0_minus_Xc;
            Vector<Scalar, Dim> new_x1_pos = xc + Rot*m_X1_minus_Xc;
            Vector<Scalar, Dim> new_x2_pos = xc + Rot*m_X2_minus_Xc;
            Vector<Scalar, Dim> new_x3_pos = xc + Rot*m_X3_minus_Xc;

            
            //need further consideration
            unsigned int x0_rigid_ele_num = this->gometric_points_[x0_index].rigidEleNum();
            new_x0_pos = (new_x0_pos + x0_rigid_ele_num*ele_vert_pos[0])/(x0_rigid_ele_num+1);
            this->gometric_points_[x0_index].setRigidEleNum(x0_rigid_ele_num+1);

            unsigned int x1_rigid_ele_num = this->gometric_points_[x1_index].rigidEleNum();
            new_x1_pos = (new_x1_pos + x1_rigid_ele_num*ele_vert_pos[1])/(x1_rigid_ele_num+1);
            this->gometric_points_[x1_index].setRigidEleNum(x1_rigid_ele_num+1);

            unsigned int x2_rigid_ele_num = this->gometric_points_[x2_index].rigidEleNum();
            new_x2_pos = (new_x2_pos + x2_rigid_ele_num*ele_vert_pos[2])/(x2_rigid_ele_num+1);
            this->gometric_points_[x2_index].setRigidEleNum(x2_rigid_ele_num+1);

            unsigned int x3_rigid_ele_num = this->gometric_points_[x3_index].rigidEleNum();
            new_x3_pos = (new_x3_pos + x3_rigid_ele_num*ele_vert_pos[3])/(x3_rigid_ele_num+1);
            this->gometric_points_[x3_index].setRigidEleNum(x3_rigid_ele_num+1);

            this->mesh_->setVertPos(x0_index, new_x0_pos);
            this->mesh_->setVertPos(x1_index, new_x1_pos);
            this->mesh_->setVertPos(x2_index, new_x2_pos);
            this->mesh_->setVertPos(x3_index, new_x3_pos);
        }
    }

    std::cout<<"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n";
    std::cout<<"rigid_ele_num: "<<rigid_ele_num<<std::endl;
    std::cout<<"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n";
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::imposeRigidConstrainByDisjoinSet()
{
    //only 3D-case is supported.
    PHYSIKA_ASSERT(Dim == 3);

    std::vector<unsigned int> rigid_ele_vec;
    for (unsigned int ele_id = 0; ele_id<this->mesh_->eleNum(); ele_id++)
    {
        Scalar ref_volume = this->pdm_base_->particle(ele_id).volume();

        std::vector<Vector<Scalar, Dim> > ele_vert_pos;
        std::vector<unsigned int> ele_vert_index;

        this->mesh_->eleVertPos(ele_id, ele_vert_pos);
        this->mesh_->eleVertIndex(ele_id, ele_vert_index);

        PHYSIKA_ASSERT(ele_vert_pos.size() == 4);
        PHYSIKA_ASSERT(ele_vert_index.size() == 4);

        Scalar ele_quality = this->computeElementQuality(ref_volume, ele_vert_pos[0], ele_vert_pos[1], ele_vert_pos[2], ele_vert_pos[3]);

        unsigned int x0_index = ele_vert_index[0];
        unsigned int x1_index = ele_vert_index[1];
        unsigned int x2_index = ele_vert_index[2];
        unsigned int x3_index = ele_vert_index[3];

        const Vector<Scalar, Dim> & m_X0_pos = this->gometric_points_[x0_index].lastVertexPos();
        const Vector<Scalar, Dim> & m_X1_pos = this->gometric_points_[x1_index].lastVertexPos();
        const Vector<Scalar, Dim> & m_X2_pos = this->gometric_points_[x2_index].lastVertexPos();
        const Vector<Scalar, Dim> & m_X3_pos = this->gometric_points_[x3_index].lastVertexPos();

        Scalar m_ele_quality = this->computeElementQuality(ref_volume, m_X0_pos, m_X1_pos, m_X2_pos, m_X3_pos);

        //need further consideration
        if (ele_quality < this->critical_ele_quality_ && ele_quality < m_ele_quality) 
            rigid_ele_vec.push_back(ele_id);
    }


    PDMDisjoinSet rigid_ele_disjoin_set;
    //make set
    rigid_ele_disjoin_set.makeSet(rigid_ele_vec);

    //union
    for (unsigned int i = 0; i < rigid_ele_vec.size(); i++)
        for (unsigned int j = i+1; j < rigid_ele_vec.size(); j++)
        {
            std::vector<unsigned int> share_vertex_vec;
            if (this->isElementConnected(rigid_ele_vec[i], rigid_ele_vec[j], share_vertex_vec) == true)
                rigid_ele_disjoin_set.unionSet(rigid_ele_vec[i], rigid_ele_vec[j]);
        }

    //find
    std::map<unsigned int, std::vector<unsigned int> > parent_map;
    for (unsigned int i = 0; i < rigid_ele_vec.size(); i++)
    {
        unsigned int parent = rigid_ele_disjoin_set.findSet(rigid_ele_vec[i]);
        parent_map[parent].push_back(rigid_ele_vec[i]);
    }


    std::cout<<"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n";
    std::cout<<"rigid_ele_num: "<<rigid_ele_vec.size()<<std::endl;
    std::cout<<"disjoin part num: "<<parent_map.size()<<std::endl;
    std::cout<<"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n";


    for (std::map<unsigned int, std::vector<unsigned int> >::const_iterator map_iter = parent_map.begin(); map_iter != parent_map.end(); map_iter++)
    {
        const std::vector<unsigned int> & all_ele_index = map_iter->second;

        //find all element vertex index belonging to current disjoin part
        std::set<unsigned int> all_ele_vert_index;
        for (unsigned int i = 0; i < all_ele_index.size(); i++)
        {
            std::vector<unsigned int> ele_vert_index;
            this->mesh_->eleVertIndex(all_ele_index[i], ele_vert_index);

            for (unsigned int j = 0; j < ele_vert_index.size(); j++)
                all_ele_vert_index.insert(ele_vert_index[j]);
        }

        std::vector<Vector<Scalar, Dim> > m_all_ele_vert_pos(all_ele_vert_index.size(), Vector<Scalar, Dim>(0.0) );
        unsigned int vert_index_id = 0;
        for (std::set<unsigned int>::iterator iter = all_ele_vert_index.begin(); iter != all_ele_vert_index.end(); iter++, vert_index_id++) 
            m_all_ele_vert_pos[vert_index_id] = this->gometric_points_[*iter].lastVertexPos();

        std::vector<Vector<Scalar, Dim> > all_ele_vert_pos(all_ele_vert_index.size(), Vector<Scalar, Dim>(0.0) );
        vert_index_id = 0;
        for (std::set<unsigned int>::iterator iter = all_ele_vert_index.begin(); iter != all_ele_vert_index.end(); iter++, vert_index_id++) 
            all_ele_vert_pos[vert_index_id] = this->mesh_->vertPos(*iter);

        Vector<Scalar, Dim> Xc(0.0);
        for (unsigned int i = 0; i<m_all_ele_vert_pos.size(); i++)
            Xc += m_all_ele_vert_pos[i];
        Xc /= m_all_ele_vert_pos.size();

        Vector<Scalar, Dim> xc(0.0);
        for (unsigned int i = 0; i<all_ele_vert_pos.size(); i++) 
            xc += all_ele_vert_pos[i];
        xc /= all_ele_vert_pos.size();

        
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Dm(Dim, all_ele_vert_index.size());
        for (unsigned int i = 0; i < all_ele_vert_index.size(); i++)
        {
            Vector<Scalar, Dim> m_col = m_all_ele_vert_pos[i] - Xc;
            for (unsigned int j = 0; j < Dim; j++)
                Dm(j, i) = m_col[j];
        }

        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Ds(Dim, all_ele_vert_index.size());
        for (unsigned int i = 0; i < all_ele_vert_index.size(); i++)
        {
            Vector<Scalar, Dim> col = all_ele_vert_pos[i] - xc;
            for (unsigned int j = 0; j < Dim; j++)
                Ds(j, i) = col[j];
        }

        //need further consideration
        Eigen::Matrix<Scalar, Dim, Dim> F = Dm.transpose().jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Ds.transpose()).transpose();

        Eigen::JacobiSVD<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> > svd(F, Eigen::ComputeThinU|Eigen::ComputeThinV);
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> U = svd.matrixU();
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> V = svd.matrixV();
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> R = U*V.transpose();

        SquareMatrix<Scalar, Dim> Rot;
        for (unsigned int i=0; i<3; i++)
            for (unsigned int j=0; j<3; j++)
                Rot(i, j) = R(i, j);


        Scalar alpha = std::acos(0.5*(Rot.trace()-1))/PI*180.0;
        if (abs(alpha) > this->max_rigid_rot_degree_)
        {
            std::cout<<"alpha: "<<alpha<<std::endl;

            alpha = alpha>0? this->max_rigid_rot_degree_:-max_rigid_rot_degree_;

            Scalar sin_alpha = std::sin(alpha/180.0*PI);
            Scalar cos_alpha = std::cos(alpha/180.0*PI);

            Vector<Scalar, Dim> rot_axis;
            rot_axis[0] = (Rot(2,1) - R(1,2))/(2*sin_alpha);
            rot_axis[1] = (Rot(0,2) - R(2,0))/(2*sin_alpha);
            rot_axis[2] = (Rot(1,0) - R(0,1))/(2*sin_alpha);
            rot_axis.normalize();

            Rot(0,0) = cos_alpha + (1-cos_alpha)*rot_axis[0]*rot_axis[0];
            Rot(0,1) = (1-cos_alpha)*rot_axis[0]*rot_axis[1] - sin_alpha*rot_axis[2];
            Rot(0,2) = (1-cos_alpha)*rot_axis[0]*rot_axis[2] + sin_alpha*rot_axis[1];

            Rot(1,0) = (1-cos_alpha)*rot_axis[1]*rot_axis[0] + sin_alpha*rot_axis[2];
            Rot(1,1) = cos_alpha + (1-cos_alpha)*rot_axis[1]*rot_axis[1];
            Rot(1,2) = (1-cos_alpha)*rot_axis[1]*rot_axis[2] - sin_alpha*rot_axis[0];

            Rot(2,0) = (1-cos_alpha)*rot_axis[2]*rot_axis[0] - sin_alpha*rot_axis[1];
            Rot(2,1) = (1-cos_alpha)*rot_axis[2]*rot_axis[1] + sin_alpha*rot_axis[0];
            Rot(2,2) = cos_alpha + (1-cos_alpha)*rot_axis[2]*rot_axis[2];

            std::cout<<"correct alpha: "<<std::acos(0.5*(Rot.trace()-1))/PI*180.0<<std::endl;
        }

        vert_index_id = 0;
        for (std::set<unsigned int>::iterator iter = all_ele_vert_index.begin(); iter != all_ele_vert_index.end(); iter++, vert_index_id++)
        {
            Vector<Scalar, Dim> new_vert_pos = xc + Rot*(m_all_ele_vert_pos[vert_index_id] - Xc);
            this->mesh_->setVertPos(*iter, new_vert_pos);
        }

    }

}

template <typename Scalar, int Dim>
bool PDMTopologyControlMethod<Scalar, Dim>::isElementConnected(unsigned int fir_ele_id, unsigned int sec_ele_id, std::vector<unsigned int> & share_vertex_vec)
{
    PHYSIKA_ASSERT(fir_ele_id != sec_ele_id);
    std::vector<unsigned int> fir_ele_vert;
    std::vector<unsigned int> sec_ele_vert;
    this->mesh_->eleVertIndex(fir_ele_id, fir_ele_vert);
    this->mesh_->eleVertIndex(sec_ele_id, sec_ele_vert);
    PHYSIKA_ASSERT(fir_ele_vert.size() == sec_ele_vert.size());
    PHYSIKA_ASSERT(fir_ele_vert.size()==3 || fir_ele_vert.size()==4);

    for (unsigned int i=0; i<fir_ele_vert.size(); i++)
        for (unsigned int j=0; j<sec_ele_vert.size(); j++)
        {
            if (fir_ele_vert[i] == sec_ele_vert[j])
                share_vertex_vec.push_back(fir_ele_vert[i]);
        }
    PHYSIKA_ASSERT(share_vertex_vec.size()<fir_ele_vert.size());
    return share_vertex_vec.size() == fir_ele_vert.size()-1;
}

template <typename Scalar, int Dim>
void PDMTopologyControlMethod<Scalar, Dim>::initEnvForRigidConstrain()
{
    for (unsigned int ver_id = 0; ver_id<this->gometric_points_.size(); ver_id++)
    {
        PHYSIKA_ASSERT(ver_id == this->gometric_points_[ver_id].vertexId());
        this->gometric_points_[ver_id].setLastVertexPos(this->mesh_->vertPos(ver_id)); //set last vertex position
        this->gometric_points_[ver_id].setRigidEleNum(0);                              //reset rigid ele num
    }
}

template <typename Scalar, int Dim>
Scalar PDMTopologyControlMethod<Scalar, Dim>::computeElementQuality(Scalar ref_volume, const Vector<Scalar, Dim> & x0, const Vector<Scalar, Dim> & x1, const Vector<Scalar, Dim> & x2, const Vector<Scalar, Dim> & x3) const
{
    Vector<Scalar, Dim> x1_minus_x0 = x1 - x0;
    Vector<Scalar, Dim> x2_minus_x0 = x2 - x0;
    Vector<Scalar, Dim> x3_minus_x0 = x3 - x0;

    Vector<Scalar, Dim> x2_minus_x1 = x2 - x1;
    Vector<Scalar, Dim> x3_minus_x2 = x3 - x2;
    Vector<Scalar, Dim> x1_minus_x3 = x1 - x3;

    //cross vector used for 2D-case compile
    Vector<Scalar, Dim> cross_vector(x1_minus_x0.cross(x2_minus_x0));
    Scalar signed_volume = 1.0/6.0*x3_minus_x0.dot(cross_vector);

    Scalar l_x1_minus_x0_square = x1_minus_x0.normSquared();
    Scalar l_x2_minus_x0_square = x2_minus_x0.normSquared();
    Scalar l_x3_minus_x0_square = x3_minus_x0.normSquared();
    Scalar l_x2_minus_x1_square = x2_minus_x1.normSquared();
    Scalar l_x3_minus_x2_square = x3_minus_x2.normSquared();
    Scalar l_x1_minus_x3_square = x1_minus_x3.normSquared();

    Scalar l_x1_minus_x0 = sqrt(l_x1_minus_x0_square);
    Scalar l_x2_minus_x0 = sqrt(l_x2_minus_x0_square);
    Scalar l_x3_minus_x0 = sqrt(l_x3_minus_x0_square);
    Scalar l_x2_minus_x1 = sqrt(l_x2_minus_x1_square);
    Scalar l_x3_minus_x2 = sqrt(l_x3_minus_x2_square);
    Scalar l_x1_minus_x3 = sqrt(l_x1_minus_x3_square);

    Scalar l_rms_square = (l_x1_minus_x0_square+l_x2_minus_x0_square+l_x3_minus_x0_square+l_x2_minus_x1_square+l_x3_minus_x2_square+l_x1_minus_x3_square)/6.0;
    Scalar l_rms_quad = l_rms_square*l_rms_square;

    Scalar reciprocal_sum = 1.0/l_x1_minus_x0 + 1.0/l_x2_minus_x0 + 1.0/l_x3_minus_x0 + 1.0/l_x2_minus_x1 + 1.0/l_x3_minus_x2 + 1.0/l_x1_minus_x3;
    Scalar l_harm = 6.0/reciprocal_sum;

    Scalar ele_quality = 8.48528137*signed_volume*l_harm/l_rms_quad;

    //Scalar vol_ratio = signed_volume/ref_volume;
    //ele_quality = min(ele_quality, vol_ratio);

    return ele_quality;
}

template <typename Scalar, int Dim>
unsigned int PDMTopologyControlMethod<Scalar, Dim>::numInvertedElement() const
{
    unsigned int inverted_num = 0;
    for (unsigned int ele_id = 0; ele_id < this->mesh_->eleNum(); ele_id++)
    {
        std::vector<Vector<Scalar, Dim> > ele_vert_pos;
        this->mesh_->eleVertPos(ele_id, ele_vert_pos);

        Vector<Scalar, Dim> x1_minus_x0 = ele_vert_pos[1] - ele_vert_pos[0];
        Vector<Scalar, Dim> x2_minus_x0 = ele_vert_pos[2] - ele_vert_pos[0];
        Vector<Scalar, Dim> x3_minus_x0 = ele_vert_pos[3] - ele_vert_pos[0];

        //cross vector used for 2D-case compile
        Vector<Scalar, Dim> cross_vector(x1_minus_x0.cross(x2_minus_x0));
        Scalar signed_volume = 1.0/6.0*x3_minus_x0.dot(cross_vector);

        if (signed_volume < 0.0) inverted_num++;
    }
    return inverted_num;
}

template <typename Scalar, int Dim>
unsigned int PDMTopologyControlMethod<Scalar, Dim>::numIsolatedElement() const
{
    unsigned int isolated_num = 0;
    unsigned int num_particle = this->pdm_base_->numSimParticles();
    for (unsigned int par_id = 0; par_id < num_particle; par_id++)
    {
        const PDMParticle<Scalar, Dim> & particle = this->pdm_base_->particle(par_id);
        if (particle.validFamilySize() == 0) isolated_num++;
    }
    return isolated_num;
}

template <typename Scalar, int Dim>
unsigned int PDMTopologyControlMethod<Scalar, Dim>::numIsolatedAndInvertedElement() const
{
    unsigned int isolated_inverted_num = 0;

    PHYSIKA_ASSERT(this->mesh_->eleNum() == this->pdm_base_->numSimParticles());
    for (unsigned int ele_id = 0; ele_id < this->mesh_->eleNum(); ele_id++)
    {
        std::vector<Vector<Scalar, Dim> > ele_vert_pos;
        this->mesh_->eleVertPos(ele_id, ele_vert_pos);

        Vector<Scalar, Dim> x1_minus_x0 = ele_vert_pos[1] - ele_vert_pos[0];
        Vector<Scalar, Dim> x2_minus_x0 = ele_vert_pos[2] - ele_vert_pos[0];
        Vector<Scalar, Dim> x3_minus_x0 = ele_vert_pos[3] - ele_vert_pos[0];

        //cross vector used for 2D-case compile
        Vector<Scalar, Dim> cross_vector(x1_minus_x0.cross(x2_minus_x0));
        Scalar signed_volume = 1.0/6.0*x3_minus_x0.dot(cross_vector);

        const PDMParticle<Scalar, Dim> & particle = this->pdm_base_->particle(ele_id);

        if (signed_volume < 0.0 && particle.validFamilySize() == 0) isolated_inverted_num++;
    }
    return isolated_inverted_num;
}

//explicit instantiations
template class PDMTopologyControlMethod<float, 2>;
template class PDMTopologyControlMethod<double, 2>;
template class PDMTopologyControlMethod<float, 3>;
template class PDMTopologyControlMethod<double, 3>;

}// end of namespace Physika