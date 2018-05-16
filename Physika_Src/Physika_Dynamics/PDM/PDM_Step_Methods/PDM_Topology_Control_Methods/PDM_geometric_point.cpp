/*
 * @file PDM_geometric_point.cpp
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

#include <numeric>
#include "Physika_Dependency/Eigen/Eigen"

#include "Physika_Dynamics/PDM/PDM_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Topology_Control_Methods/PDM_geometric_point.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Topology_Control_Methods/PDM_topology_control_method.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Topology_Control_Methods/PDM_element_tuple.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Topology_Control_Methods/PDM_disjoin_set.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/tet_mesh.h"
#include "Physika_Core/Utilities/physika_assert.h"

namespace Physika{

template <typename Scalar, int Dim>
PDMGeometricPoint<Scalar, Dim>::PDMGeometricPoint()
    :vertex_id_(0), potential_split_(false), need_crack_adjust_(false),
    last_vertex_pos_(0.0), rigid_ele_num_(0)
{

}

template <typename Scalar, int Dim>
PDMGeometricPoint<Scalar, Dim>::~PDMGeometricPoint()
{

}

template <typename Scalar, int Dim>
unsigned int PDMGeometricPoint<Scalar, Dim>::vertexId() const
{
    return this->vertex_id_;
}

template <typename Scalar, int Dim>
bool PDMGeometricPoint<Scalar, Dim>::isPotentialSplit() const
{
    return this->potential_split_;
}

template <typename Scalar, int Dim>
unsigned int PDMGeometricPoint<Scalar, Dim>::adjoinNum() const
{
    return this->adjoin_elements_.size();
}

template <typename Scalar, int Dim>
const std::set<unsigned int> & PDMGeometricPoint<Scalar, Dim>::adjoinElement()
{
    return this->adjoin_elements_;
}

template <typename Scalar, int Dim>
const Vector<Scalar, Dim> & PDMGeometricPoint<Scalar, Dim>::lastVertexPos() const
{
    return this->last_vertex_pos_;
}

template <typename Scalar, int Dim>
unsigned int PDMGeometricPoint<Scalar, Dim>::rigidEleNum() const
{
    return this->rigid_ele_num_;
}

template <typename Scalar, int Dim>
void PDMGeometricPoint<Scalar, Dim>::setVertexId(unsigned int vertex_id)
{
    this->vertex_id_ = vertex_id;
}

template <typename Scalar, int Dim>
void PDMGeometricPoint<Scalar, Dim>::setPotentialSplit(bool potential_split)
{
    this->potential_split_ = potential_split;
}

template <typename Scalar, int Dim>
void PDMGeometricPoint<Scalar, Dim>::setLastVertexPos(const Vector<Scalar, Dim> & last_vertex_pos)
{
    this->last_vertex_pos_ = last_vertex_pos;
}

template <typename Scalar, int Dim>
void PDMGeometricPoint<Scalar, Dim>::setRigidEleNum(unsigned int rigid_ele_num)
{
    this->rigid_ele_num_ = rigid_ele_num;
}

template <typename Scalar, int Dim>
void PDMGeometricPoint<Scalar, Dim>::addCrackElementTuple(const PDMElementTuple & crack_ele_tuple)
{
    std::pair<std::set<PDMElementTuple>::iterator, bool> ret_pair;
    ret_pair = this->crack_element_tuples_.insert(crack_ele_tuple);
    if (ret_pair.second == false)
    {
        std::cerr<<"error: crack face is already existed, can't insert it!\n";
        std::exit(EXIT_FAILURE);
    }

    //set vertex to pentially split
    this->potential_split_ = true;
}

template <typename Scalar, int Dim>
void PDMGeometricPoint<Scalar, Dim>::clearCrackElementTuple()
{
    this->crack_element_tuples_.clear();
}

template <typename Scalar, int Dim>
void PDMGeometricPoint<Scalar, Dim>::addAdjoinElement(unsigned int ele_id)
{
    this->adjoin_elements_.insert(ele_id);
}

template <typename Scalar, int Dim>
void PDMGeometricPoint<Scalar, Dim>::clearAdjoinElement()
{
    this->adjoin_elements_.clear();
}

template <typename Scalar, int Dim>
bool PDMGeometricPoint<Scalar, Dim>::splitByDisjoinSet(PDMTopologyControlMethod<Scalar, Dim> * topology_control_method)
{
    //skip unnecessary splitting
    if (this->potential_split_ == false) return false;

    //set potential_split_ to false
    this->potential_split_ = false;

    //element disjoin set
    PDMDisjoinSet element_disjoin_set;

    //make set
    element_disjoin_set.makeSet(this->adjoin_elements_);

    //union
    VolumetricMesh<Scalar, Dim> * mesh = topology_control_method->mesh();
    for (std::set<unsigned int>::const_iterator i_iter = this->adjoin_elements_.begin(); i_iter != this->adjoin_elements_.end(); i_iter++)
        for (std::set<unsigned int>::const_iterator j_iter = i_iter; j_iter != this->adjoin_elements_.end(); j_iter++)
        {
            if (j_iter != i_iter && this->isElementsConnected(mesh, *i_iter, *j_iter) == true)
            {
                element_disjoin_set.unionSet(*i_iter, *j_iter);
            }
        }

    //find
    std::map<unsigned int, std::vector<unsigned int> > parent_map;
    for (std::set<unsigned int>::const_iterator iter = this->adjoin_elements_.begin(); iter != this->adjoin_elements_.end(); iter++)
    {
        unsigned int parent = element_disjoin_set.findSet(*iter);
        parent_map[parent].push_back(*iter);
    }

    // if size of parent_map is greater than 1, then it means that geometric point need split
    if (parent_map.size()>1)
    {
        //need adjustment in next time step
        this->need_crack_adjust_ = true;

        //delete crack element tuple
        for (std::set<PDMElementTuple>::const_iterator ele_tuple_iter = this->crack_element_tuples_.begin(); ele_tuple_iter != this->crack_element_tuples_.end(); ele_tuple_iter++)
        {
            const std::vector<unsigned int> & ele_tuple_vec = ele_tuple_iter->eleVec();
            PHYSIKA_ASSERT(ele_tuple_vec.size() == 2);
            if (element_disjoin_set.findSet(ele_tuple_vec[0]) != element_disjoin_set.findSet(ele_tuple_vec[1]))
            {
                topology_control_method->deleteElementTuple(*ele_tuple_iter);
            }
        }

        //update adjoin elements
        this->adjoin_elements_.clear();
        const std::map<unsigned int, std::vector<unsigned int> >::const_iterator first_iter = parent_map.begin();
        const std::vector<unsigned int> & first_adjoin_elements = first_iter->second;
        this->adjoin_elements_.insert(first_adjoin_elements.begin(), first_adjoin_elements.end());

        //note: pointer "this" may be invalid during loop
        unsigned int vertex_id = this->vertex_id_;

        //add vertices into mesh and geometric points into topology control method
        std::map<unsigned int, std::vector<unsigned int> >::iterator map_iter = ++ parent_map.begin();
        for ( ; map_iter != parent_map.end(); map_iter++)
        {
            //add vertex into mesh
            mesh->addVertex(mesh->vertPos(vertex_id));

            //define another geometric point
            PDMGeometricPoint<Scalar, Dim> another_geometric_point;
            another_geometric_point.vertex_id_ = mesh->vertNum()-1;

            //need adjustment in next time step
            another_geometric_point.need_crack_adjust_ = true;

            //insert adjoin elements
            const std::vector<unsigned int> & another_adjoin_elements = map_iter->second;
            for(unsigned int adjoin_ele_id = 0; adjoin_ele_id < another_adjoin_elements.size(); adjoin_ele_id++)
                another_geometric_point.addAdjoinElement(another_adjoin_elements[adjoin_ele_id]);

            //add topology change
            const std::vector<unsigned int> & ele_vec = map_iter->second;
            for (unsigned int ele_id = 0; ele_id<ele_vec.size(); ele_id++)
            {
                std::vector<unsigned int> topology_change;
                topology_change.push_back(ele_vec[ele_id]);                    //ele_id
                topology_change.push_back(vertex_id);                          //old vertex id
                topology_change.push_back(another_geometric_point.vertex_id_); //new global vertex id
                topology_control_method->addTopologyChange(topology_change);   //add topology change to control method
            }

            //add another geometric point to topology control method
            topology_control_method->addGeometricPoint(another_geometric_point);
        }

        return true;
    }

    return false;
}

template <typename Scalar, int Dim>
void PDMGeometricPoint<Scalar, Dim>::updateGeometricPointPos(PDMTopologyControlMethod<Scalar, Dim> * topology_control_method, Scalar dt)
{
    PDMBase<Scalar, Dim> * pdm_base = topology_control_method->driver();
    VolumetricMesh<Scalar, Dim> * mesh = topology_control_method->mesh();
    Vector<Scalar, Dim> vert_pos =  mesh->vertPos(this->vertex_id_);

    if (this->adjoin_elements_.size() > 1)
    {
        Vector<Scalar, Dim> total_weighted_vel(0.0);
        Scalar total_weight = 0.0;
        for (std::set<unsigned int>::const_iterator iter = this->adjoin_elements_.begin(); iter != this->adjoin_elements_.end(); iter++)
        {
            const PDMParticle<Scalar, Dim> & particle = pdm_base->particle(*iter);
            Scalar weight = particle.mass();

            //Vector<Scalar, Dim>(0.0) used to compile 2D case
            Vector<Scalar, Dim> rot_vel = Vector<Scalar, Dim>(0.0) + topology_control_method->rotateVelocity(*iter).cross(vert_pos - pdm_base->particleCurrentPosition(*iter));

            total_weighted_vel += weight*(pdm_base->particleVelocity(*iter) + rot_vel);
            total_weight += weight;
        }

        Vector<Scalar, Dim> average_vel = total_weighted_vel/total_weight;

        Vector<Scalar, Dim> new_vert_pos = vert_pos + average_vel*dt;
        mesh->setVertPos(this->vertex_id_, new_vert_pos);
    }
    else if(this->adjoin_elements_.size() == 1)
    {
        std::set<unsigned int>::const_iterator iter = this->adjoin_elements_.begin();
        Vector<Scalar, Dim> central_pos = pdm_base->particleCurrentPosition(*iter);

        Vector<Scalar, Dim> relative_pos = vert_pos - central_pos;
        Scalar relative_pos_norm = relative_pos.norm();

        Vector<Scalar, Dim> unit_relative_pos = relative_pos.normalize();
        Vector<Scalar, Dim> rot_vel = topology_control_method->rotateVelocity(*iter);
        unit_relative_pos += rot_vel.cross(unit_relative_pos)*dt;
        unit_relative_pos.normalize();

        Vector<Scalar, Dim> trans_vel = pdm_base->particleVelocity(*iter);

        Vector<Scalar, Dim> new_vert_pos = central_pos + trans_vel*dt + unit_relative_pos*relative_pos_norm;
        mesh->setVertPos(this->vertex_id_, new_vert_pos);
    }
    else
    {
        std::cerr<<"error: geometric point has no adjoin element!\n";
        std::exit(EXIT_FAILURE);
    }
}

template <typename Scalar, int Dim>
bool PDMGeometricPoint<Scalar, Dim>::smoothCrackGeometricPointPos(PDMTopologyControlMethod<Scalar, Dim> * topology_control_method)
{
    if (this->need_crack_adjust_ == false) return false;
    this->need_crack_adjust_ = false;

    if (this->adjoin_elements_.size() <= 1) return false;

    VolumetricMesh<Scalar, Dim> * mesh = topology_control_method->mesh();

    Vector<Scalar, Dim> total_pos(0.0);
    unsigned int total_num = 0;
    for (std::set<unsigned int>::const_iterator iter = this->adjoin_elements_.begin(); iter != this->adjoin_elements_.end(); iter++)
    {
        std::vector<unsigned int> ele_index_vec;
        mesh->eleVertIndex(*iter, ele_index_vec);
        PHYSIKA_ASSERT(ele_index_vec.size() == 3 ||  ele_index_vec.size() == 4);

        for (unsigned int i=0; i<ele_index_vec.size(); i++)
        {
            if (mesh->isBoundaryVertex(ele_index_vec[i]))
            {
                total_pos += mesh->vertPos(ele_index_vec[i]);
                total_num++;
            }
        }
    }

    Vector<Scalar, Dim> average_pos = total_pos/total_num;
    Vector<Scalar, Dim> vert_pos = mesh->vertPos(this->vertex_id_);
    Scalar smooth_crack_level = topology_control_method->crackSmoothLevel();
    Vector<Scalar, Dim> new_vert_pos = vert_pos + smooth_crack_level*(average_pos - vert_pos);
    mesh->setVertPos(this->vertex_id_, new_vert_pos);

    return true;
}

template <typename Scalar, int Dim>
bool PDMGeometricPoint<Scalar, Dim>::isElementsConnected(const VolumetricMesh<Scalar, Dim> * mesh, unsigned int fir_ele_id, unsigned int sec_ele_id)
{
    PHYSIKA_ASSERT(fir_ele_id != sec_ele_id);

    PDMElementTuple ele_tuple;
    ele_tuple.setElementVec(fir_ele_id, sec_ele_id);

    if(this->crack_element_tuples_.count(ele_tuple) == 1)
        return false;

    std::vector<unsigned int> fir_ele_vert;
    std::vector<unsigned int> sec_ele_vert;
    mesh->eleVertIndex(fir_ele_id, fir_ele_vert);
    mesh->eleVertIndex(sec_ele_id, sec_ele_vert);
    PHYSIKA_ASSERT(fir_ele_vert.size() == sec_ele_vert.size());
    PHYSIKA_ASSERT(fir_ele_vert.size()==3 || fir_ele_vert.size()==4);

    unsigned int share_vertex_num = 0;
    for (unsigned int i=0; i<fir_ele_vert.size(); i++)
        for (unsigned int j=0; j<sec_ele_vert.size(); j++)
        {
            if (fir_ele_vert[i] == sec_ele_vert[j])
                share_vertex_num++;
        }
    PHYSIKA_ASSERT(share_vertex_num < fir_ele_vert.size());

    if (share_vertex_num == fir_ele_vert.size()-1)
        return true;
    else
        return false;
}

template <typename Scalar, int Dim>
Vector<Scalar, Dim> PDMGeometricPoint<Scalar, Dim>::calculateAverageVelocity(PDMTopologyControlMethod<Scalar, Dim> * topology_control_method)
{
    PHYSIKA_ASSERT(this->adjoin_elements_.size()>0);
    PDMBase<Scalar, Dim> * pdm_base = topology_control_method->driver();
    Vector<Scalar, Dim> total_vel(0.0);
    for (std::set<unsigned int>::const_iterator iter = this->adjoin_elements_.begin(); iter != this->adjoin_elements_.end(); iter++)
        total_vel += pdm_base->particleVelocity(*iter);
    return total_vel/this->adjoin_elements_.size();
}

// explicit instantiations
template class PDMGeometricPoint<float, 2>;
template class PDMGeometricPoint<double, 2>;
template class PDMGeometricPoint<float, 3>;
template class PDMGeometricPoint<double, 3>;

}//end of namespace Physika