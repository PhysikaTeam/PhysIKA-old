/*
 * @file PDM_collision_method_space_hash_3d.cpp
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

#include <iostream>
#include <fstream>

#include "Physika_Core/Timer/timer.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Vectors/vector_3d.h"

#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"
#include "Physika_Geometry/Geometry_Intersections/tetrahedron_tetrahedron_intersection.h"
#include "Physika_Dynamics/PDM/PDM_base.h"
#include "Physika_Dynamics/PDM/PDM_Utility_Set/PDM_utility_set.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Collision_Methods/PDM_collision_method_space_hash_3d.h"


namespace Physika{

template <typename Scalar>
PDMCollisionMethodSpaceHash<Scalar, 3>::PDMCollisionMethodSpaceHash()
    :grid_cell_size_(0.1), use_edge_intersect_(false), use_overlap_vol_collision_response_(true)
{

}

template <typename Scalar>
PDMCollisionMethodSpaceHash<Scalar, 3>::~PDMCollisionMethodSpaceHash()
{

}

template <typename Scalar>
Scalar PDMCollisionMethodSpaceHash<Scalar, 3>::gridCellSize() const
{
    return this->grid_cell_size_;
}

template <typename Scalar>
unsigned int PDMCollisionMethodSpaceHash<Scalar, 3>::hashTableSize() const
{
    return this->element_hash_table_.size();
}

template <typename Scalar>
void PDMCollisionMethodSpaceHash<Scalar, 3>::enableEdgeIntersect()
{
    this->use_edge_intersect_ = true;
}

template <typename Scalar>
void PDMCollisionMethodSpaceHash<Scalar, 3>::disableEdgeIntersect()
{
    this->use_edge_intersect_ = false;
}

template <typename Scalar>
void PDMCollisionMethodSpaceHash<Scalar, 3>::enableOverlapVolCollisionResponse()
{
    this->use_overlap_vol_collision_response_ = true;
}

template <typename Scalar>
void PDMCollisionMethodSpaceHash<Scalar, 3>::disableOverlapVolCollisionResponse()
{
    this->use_overlap_vol_collision_response_ = false;
}

template <typename Scalar>
void PDMCollisionMethodSpaceHash<Scalar, 3>::collisionMethod()
{
    //need further consideration
    /////////////////////////////////////////////////////////////////////////////////////////
    std::fstream file("Kc.txt", std::ios::in);
    if (file.fail() == false)
        file>>this->Kc_;
    file.close();
    std::cout<<"collision Kc: "<<this->Kc_<<std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////

    Timer timer;

    timer.startTimer();
    //reset hash table
    this->resetHashTable();
    //refresh pre stored mesh information
    this->refreshPreStoredMeshInfomation();
    timer.stopTimer();

    std::cout<<"--------------------------------------------------------------------"<<std::endl;
    std::cout<<"initialize time: "<<timer.getElapsedTime()<<std::endl;

    //initialize hash table
    timer.startTimer();
    if (use_edge_intersect_ == false)
    {
        locateVertexToHashTable();
        locateElementToHashTable();
    }
    else
    {
        locateElementToHashTable();
    }
    timer.stopTimer();

    std::cout<<"locate time: "<<timer.getElapsedTime()<<std::endl;
    std::cout<<"--------------------------------------------------------------------"<<std::endl;

    timer.startTimer();
    if (use_edge_intersect_ == false) 
        this->collisionDetectionAndResponseViaVertexPenetrate();
    else 
        this->collisionDetectionAndResponseViaEdgeIntersect();
    timer.stopTimer();

    std::cout<<"--------------------------------------------------------------------"<<std::endl;
    std::cout<<"detection and response time: "<<timer.getElapsedTime()<<std::endl;
    std::cout<<"--------------------------------------------------------------------"<<std::endl;

}

template <typename Scalar>
void PDMCollisionMethodSpaceHash<Scalar, 3>::setHashTableSize(unsigned int hash_table_size)
{
    if(use_edge_intersect_ == false)
    {
        this->vertex_hash_table_.clear();
        this->vertex_hash_table_.resize(hash_table_size, std::vector<unsigned int>());
    }
    
    this->element_hash_table_.clear();
    this->element_hash_table_.resize(hash_table_size, std::vector<unsigned int>());
}

template <typename Scalar>
void PDMCollisionMethodSpaceHash<Scalar, 3>::setGridCellSize(Scalar grid_cell_size)
{
    PHYSIKA_ASSERT(grid_cell_size > 0);
    this->grid_cell_size_ = grid_cell_size;
}

template <typename Scalar>
void PDMCollisionMethodSpaceHash<Scalar, 3>::autoSetGridCellSize()
{
    PHYSIKA_ASSERT(driver_);
    VolumetricMesh<Scalar, 3> * mesh = driver_->mesh();
    PHYSIKA_ASSERT(mesh);

    PHYSIKA_ASSERT(mesh->elementType() == VolumetricMeshInternal::TET);

    double total_edge_len = 0;
    for (unsigned int ele_id = 0; ele_id < mesh->eleNum(); ele_id++)
    {
        std::vector<Vector<Scalar, 3> > ele_pos_vec;
        mesh->eleVertPos(ele_id, ele_pos_vec);

        for (unsigned int i = 0; i < 4; i++)
        for (unsigned int j = i+1; j < 4; j++)
        {
            Vector<Scalar, 3> edge = ele_pos_vec[j] - ele_pos_vec[i];
            total_edge_len += edge.norm();
        }
    }

    //need further consideration
    Scalar average_len = total_edge_len/(6*mesh->eleNum());
    this->grid_cell_size_ = 1.1*average_len;
}

template <typename Scalar>
void PDMCollisionMethodSpaceHash<Scalar, 3>::resetHashTable()
{
    if (use_edge_intersect_ == false)
    {
        #pragma omp parallel for
        for (long long i = 0; i < this->vertex_hash_table_.size(); i++)
            this->vertex_hash_table_[i].clear();
    }

    #pragma omp parallel for
    for (long long i = 0; i < this->vertex_hash_table_.size(); i++)
        this->element_hash_table_[i].clear();
}


template <typename Scalar>
void PDMCollisionMethodSpaceHash<Scalar, 3>::locateVertexToHashTable()
{
    PHYSIKA_ASSERT(driver_);
    VolumetricMesh<Scalar, 3> * mesh = driver_->mesh();
    PHYSIKA_ASSERT(mesh);

    #pragma omp parallel for
    for (long long ver_id = 0; ver_id < mesh->vertNum(); ver_id++)
    {
        Vector<Scalar, 3> ver_pos = mesh->vertPos(ver_id);
        long long i = ver_pos[0]/this->grid_cell_size_;
        long long j = ver_pos[1]/this->grid_cell_size_;
        long long k = ver_pos[2]/this->grid_cell_size_;

        unsigned int hash_pos = this->hashFunction(i, j, k);
        #pragma omp critical(LOCATE_VERTEX)
        this->vertex_hash_table_[hash_pos].push_back(ver_id);
    }
}

template <typename Scalar>
void PDMCollisionMethodSpaceHash<Scalar, 3>::locateElementToHashTable()
{
    PHYSIKA_ASSERT(driver_);
    VolumetricMesh<Scalar, 3> * mesh = driver_->mesh();
    PHYSIKA_ASSERT(mesh);

    #pragma omp parallel for
    for (long long ele_id = 0; ele_id < mesh->eleNum(); ele_id++)
    {
        std::vector<Vector<Scalar, 3> > ele_pos_vec;
        mesh->eleVertPos(ele_id, ele_pos_vec);

        Vector<Scalar, 3> min_corner = PDMUtilitySet<Scalar, 3>::computeMinCorner(ele_pos_vec);
        Vector<Scalar, 3> max_corner = PDMUtilitySet<Scalar, 3>::computeMaxCorner(ele_pos_vec);

        //update bounding volume information
        this->mesh_ele_min_corner_vec_[ele_id] = min_corner;
        this->mesh_ele_max_corner_vec_[ele_id] = max_corner;

        long long hash_x_pos_min = min_corner[0]/this->grid_cell_size_;
        long long hash_y_pos_min = min_corner[1]/this->grid_cell_size_;
        long long hash_z_pos_min = min_corner[2]/this->grid_cell_size_;

        long long hash_x_pos_max = max_corner[0]/this->grid_cell_size_;
        long long hash_y_pos_max = max_corner[1]/this->grid_cell_size_;
        long long hash_z_pos_max = max_corner[2]/this->grid_cell_size_;

        for (long x_hash_idx = hash_x_pos_min; x_hash_idx <= hash_x_pos_max; x_hash_idx++)
        for (long y_hash_idx = hash_y_pos_min; y_hash_idx <= hash_y_pos_max; y_hash_idx++)
        for (long z_hash_idx = hash_z_pos_min; z_hash_idx <= hash_z_pos_max; z_hash_idx++)
        {
            unsigned int hash_pos = this->hashFunction(x_hash_idx, y_hash_idx, z_hash_idx);
            #pragma omp critical(LOCATE_ELEMENT)
            this->element_hash_table_[hash_pos].push_back(ele_id);
        }
    }

}

template <typename Scalar>
void PDMCollisionMethodSpaceHash<Scalar, 3>::collisionDetectionAndResponseViaVertexPenetrate()
{
    PHYSIKA_ASSERT(driver_);
    VolumetricMesh<Scalar, 3> * mesh = driver_->mesh();
    PHYSIKA_ASSERT(mesh);
    PHYSIKA_ASSERT(mesh->elementType() == VolumetricMeshInternal::TET);

    //generate vertex adjoin element vector
    std::vector<std::vector<unsigned int> >  vert_adjoin_ele_vec;
    this->generateVertexAdjoinElement(mesh, vert_adjoin_ele_vec);

    //calculate A_inverse_vec
    std::vector<SquareMatrix<Scalar, 3> > A_inverse_vec(mesh->eleNum(), SquareMatrix<Scalar, 3>(0.0));

    #pragma omp parallel for
    for (long long ele_id = 0; ele_id < mesh->eleNum(); ele_id++)
    {
        std::vector<Vector<Scalar, 3> > & ele_vert_pos = this->mesh_ele_pos_vec_[ele_id];

        SquareMatrix<Scalar, 3> A(0.0);
        for (unsigned int n = 0; n < 3; n++)
        for (unsigned int m = 0; m < 3; m++)
            A(n, m) = ele_vert_pos[m+1][n] - ele_vert_pos[0][n];

        // P = X0 +A*beta ==> beta = A^(-1)*(P-X0)
        A_inverse_vec[ele_id] = A.inverse();
    }

    std::set<std::pair<unsigned int, unsigned int> > collision_result;

    //collision detection
    PHYSIKA_ASSERT(this->vertex_hash_table_.size() == this->element_hash_table_.size());
    #pragma omp parallel for
    for (long long hash_idx = 0; hash_idx < this->element_hash_table_.size(); hash_idx++)
        for (unsigned int ele_idx = 0; ele_idx < this->element_hash_table_[hash_idx].size(); ele_idx++)
        {
            unsigned int ele_id = this->element_hash_table_[hash_idx][ele_idx];
            const Vector<Scalar, 3> & x0 = this->mesh_ele_pos_vec_[ele_id][0]; 

            for (unsigned int vert_idx = 0; vert_idx < this->vertex_hash_table_[hash_idx].size(); vert_idx++)
            {
                unsigned int vert_id = this->vertex_hash_table_[hash_idx][vert_idx];

                bool result_via_vertex_penetrate = this->intersectTethedraViaVertexPenetrate(ele_id, vert_id, A_inverse_vec[ele_id], x0);

                if (result_via_vertex_penetrate == true)
                {
                    for (unsigned int vert_adjoin_idx = 0; vert_adjoin_idx < vert_adjoin_ele_vec[vert_id].size(); vert_adjoin_idx++)
                    {
                        unsigned int collision_ele_id = vert_adjoin_ele_vec[vert_id][vert_adjoin_idx];

                        //////////////////////////////////////////////////////////////////////////////////
                        unsigned int fir_ele_id = ele_id;
                        unsigned int sec_ele_id = collision_ele_id;
                        const std::vector<unsigned int> & fir_ele_vert = this->mesh_ele_index_vec_[fir_ele_id];
                        const std::vector<unsigned int> & sec_ele_vert = this->mesh_ele_index_vec_[sec_ele_id];
                        PHYSIKA_ASSERT(fir_ele_vert.size() == sec_ele_vert.size());
                        PHYSIKA_ASSERT(fir_ele_vert.size()==3 || fir_ele_vert.size()==4);

                        unsigned int shared_num = 0;
                        for (unsigned int i=0; i<fir_ele_vert.size(); i++)
                            for (unsigned int j=0; j<sec_ele_vert.size(); j++)
                            {
                                if (fir_ele_vert[i] == sec_ele_vert[j])
                                    shared_num++;
                            }
                        if (shared_num >= 1) continue;
                        //////////////////////////////////////////////////////////////////////////////////

                        std::pair<unsigned int, unsigned int> collision_pair(ele_id, collision_ele_id);
                        if (collision_pair.first > collision_pair.second) std::swap(collision_pair.first, collision_pair.second);

                        #pragma omp critical(COLLISION_RESULT)
                        collision_result.insert(collision_pair);
                    }
                }

            }
        }


    //collision response
    this->collisionResponse(collision_result);

    std::cout<<"**********************************************\n";
    std::cout<<"collision num: "<<collision_result.size()<<std::endl;
    std::cout<<"**********************************************\n";
}

template <typename Scalar>
void PDMCollisionMethodSpaceHash<Scalar, 3>::collisionDetectionAndResponseViaEdgeIntersect()
{
    PHYSIKA_ASSERT(driver_);
    VolumetricMesh<Scalar, 3> * mesh = driver_->mesh();
    PHYSIKA_ASSERT(mesh);
    PHYSIKA_ASSERT(mesh->elementType() == VolumetricMeshInternal::TET);

    //generate element face normal
    Timer timer;
    timer.startTimer();
    std::vector<std::vector<Vector<Scalar, 3> > > ele_face_normal_vec;
    this->generateElementFaceNormal(mesh, ele_face_normal_vec);
    timer.stopTimer();

    std::cout<<"--------------------------------------------------------------------"<<std::endl;
    std::cout<<"time cost for generate face normal: "<<timer.getElapsedTime()<<std::endl;
    std::cout<<"--------------------------------------------------------------------"<<std::endl;

    std::set<std::pair<unsigned int, unsigned int> > collision_result;

    std::vector<unsigned char>        global_masks(4);
    std::vector<std::vector<Scalar> > global_coord(4, std::vector<Scalar>(4));
    std::vector<Vector<Scalar,3> >    global_teta_to_tetb_vec(4);
    std::vector<Vector<Scalar,3> >    global_tetb_to_teta_vec(4); 

    //collision detection
    #pragma omp parallel for firstprivate(global_masks, global_coord, global_teta_to_tetb_vec, global_tetb_to_teta_vec)
    for (long long hash_idx = 0; hash_idx < this->element_hash_table_.size(); hash_idx++)
    {
        unsigned int hash_bin_size = this->element_hash_table_[hash_idx].size();
        for (unsigned int i = 0; i < hash_bin_size; i++)
        for (unsigned int j = i+1; j < hash_bin_size; j++)
        {
            unsigned int par_i = this->element_hash_table_[hash_idx][i];
            unsigned int par_j = this->element_hash_table_[hash_idx][j];

            std::pair<unsigned int, unsigned int> collision_pair(par_i, par_j);
            if (collision_pair.first > collision_pair.second) std::swap(collision_pair.first, collision_pair.second);

            //#pragma omp critical(COLLSION_RESULT)
            //if (collision_result.count(collision_pair) == 1) continue;

            bool result_via_edge_intersect = this->intersectTetrahedraViaEdgeIntersect(par_i, par_j, ele_face_normal_vec[par_i], ele_face_normal_vec[par_j], 
                                                                                       global_masks, global_coord, global_teta_to_tetb_vec, global_tetb_to_teta_vec);
            if (result_via_edge_intersect == true)
            {
                #pragma omp critical(COLLISION_RESULT)
                collision_result.insert(collision_pair);
            }
        }
    }

    //collision response
    this->collisionResponse(collision_result);

    std::cout<<"**********************************************\n";
    std::cout<<"collision num: "<<collision_result.size()<<std::endl;
    std::cout<<"**********************************************\n";

}

template <typename Scalar>
unsigned int PDMCollisionMethodSpaceHash<Scalar, 3>::hashFunction(long long i, long long j, long long k) const
{
    long long temp = ((i*73856093)^(j*19349663)^(k*83492791))%this->element_hash_table_.size();
    return (temp + this->element_hash_table_.size())%this->element_hash_table_.size();
}

template <typename Scalar>
void PDMCollisionMethodSpaceHash<Scalar, 3>::collisionResponse(const std::set<std::pair<unsigned int, unsigned int> > & collision_result)
{
    std::vector<std::pair<unsigned int, unsigned int> > collision_result_vec;

    for (std::set<std::pair<unsigned int, unsigned int> >::iterator iter = collision_result.begin(); iter != collision_result.end(); iter++)
        collision_result_vec.push_back(*iter);

    #pragma omp parallel for
    for (long long idx = 0; idx < collision_result_vec.size(); idx++)
    {
        unsigned int par_i = collision_result_vec[idx].first;
        unsigned int par_j = collision_result_vec[idx].second;

        Vector<Scalar, 3> unit_relative_pos = driver_->particleCurrentPosition(par_i) - driver_->particleCurrentPosition(par_j);
        unit_relative_pos.normalize();

        //need further consideration
        const Vector<Scalar, 3> relative_vel = driver_->particleVelocity(par_j) - driver_->particleVelocity(par_i);
        Scalar proj_relative_vel_norm = relative_vel.dot(unit_relative_pos);
        if (proj_relative_vel_norm < 0.0) proj_relative_vel_norm = -proj_relative_vel_norm/2;

        const PDMParticle<Scalar, 3> & particle_i = driver_->particle(par_i);
        const PDMParticle<Scalar, 3> & particle_j = driver_->particle(par_j);

        Scalar overlap_vol = (particle_i.volume() + particle_j.volume())/2;
        if (use_overlap_vol_collision_response_)
            overlap_vol = this->tetrahedraOverlapVolume(par_i, par_j);

        Vector<Scalar, 3> force = Kc_*overlap_vol*unit_relative_pos;
        //Vector<Scalar, 3> force = Kc_*overlap_vol*proj_relative_vel_norm*unit_relative_pos;

        // add force, omp atomic for Vector<Scalar,3> is used to prevent competition
        this->driver_->addParticleForce(par_i, force);
        this->driver_->addParticleForce(par_j, -force);
    }
}

template <typename Scalar>
void PDMCollisionMethodSpaceHash<Scalar, 3>::refreshPreStoredMeshInfomation()
{
    VolumetricMesh<Scalar, 3> * mesh = this->driver_->mesh();
    PHYSIKA_ASSERT(mesh);
    PHYSIKA_ASSERT(mesh->eleNum() == this->driver_->numSimParticles());

    //resize if not initialized
    if ( this->mesh_ele_index_vec_.size() != mesh->eleNum() || this->mesh_ele_pos_vec_.size() != mesh->eleNum() ||
         this->mesh_ele_min_corner_vec_.size() != mesh->eleNum() || this->mesh_ele_max_corner_vec_.size() != mesh->eleNum())
    {
        this->mesh_ele_index_vec_.resize(mesh->eleNum());
        this->mesh_ele_pos_vec_.resize(mesh->eleNum());
        this->mesh_ele_min_corner_vec_.resize(mesh->eleNum());
        this->mesh_ele_max_corner_vec_.resize(mesh->eleNum());
    }

    #pragma omp parallel for
    for (long long ele_id = 0; ele_id < mesh->eleNum(); ele_id++)
    {
        //element vert index
        std::vector<unsigned int> ele_vert;
        mesh->eleVertIndex(ele_id, ele_vert);
        this->mesh_ele_index_vec_[ele_id] = ele_vert;

        //element vert pos
        std::vector<Vector<Scalar, 3> > ele_pos_vec;
        mesh->eleVertPos(ele_id, ele_pos_vec);
        this->mesh_ele_pos_vec_[ele_id] = ele_pos_vec;
    }
}

template <typename Scalar>
void PDMCollisionMethodSpaceHash<Scalar, 3>::generateVertexAdjoinElement(const VolumetricMesh<Scalar, 3> * mesh, std::vector<std::vector<unsigned int> > & vert_adjoin_ele_vec)
{
    PHYSIKA_ASSERT(mesh);
    vert_adjoin_ele_vec.clear();
    vert_adjoin_ele_vec.resize(mesh->vertNum(), std::vector<unsigned int>());

    //add adjoin elements to vertex
    unsigned int ele_num = mesh->eleNum();
    #pragma omp parallel for
    for (long long ele_id = 0; ele_id < ele_num; ele_id++)
    {
        std::vector<unsigned int> ele_vert_index;
        mesh->eleVertIndex(ele_id, ele_vert_index);

        for (unsigned int vert_id = 0; vert_id < ele_vert_index.size(); vert_id++)
            #pragma omp critical(INSERT_VERT_ADJOIN_ELE)
            vert_adjoin_ele_vec[ele_vert_index[vert_id]].push_back(ele_id);
    }
}

template <typename Scalar>
bool PDMCollisionMethodSpaceHash<Scalar, 3>::isVertexPenetrateElement(const SquareMatrix<Scalar, 3> & A_inverse, const Vector<Scalar, 3> & x0, const Vector<Scalar, 3> & p) const
{
    Vector<Scalar, 3> p_minus_x0 = p - x0;
    // P = X0 +A*beta ==> beta = A^(-1)*(P-X0)
    Vector<Scalar, 3> beta = A_inverse*p_minus_x0;

    // (+, -)1.0e-10 is used to avoid numerical error
    if (beta[0] >= -1.0e-10 && beta[1] >= -1.0e-10 && beta[2] >= -1.0e-10 && beta[0]+beta[1]+beta[2] <= 1.0 + 1.0e-10) 
        return true;

    return false;
}

template <typename Scalar>
bool PDMCollisionMethodSpaceHash<Scalar, 3>::isElementContainVertex(unsigned int ele_id, unsigned int vert_id) const
{
    const std::vector<unsigned int> & ele_vert_index = this->mesh_ele_index_vec_[ele_id];
    for (unsigned int i = 0; i<ele_vert_index.size(); i++)
    {
        if (ele_vert_index[i] == vert_id)
            return true;
    }
    return false;
}

template <typename Scalar>
bool PDMCollisionMethodSpaceHash<Scalar, 3>::isVertexInsideBoundingVolume(unsigned int ele_id, const Vector<Scalar, 3> & vert_pos) const
{
    const Vector<Scalar, 3> & min_corner = this->mesh_ele_min_corner_vec_[ele_id];
    const Vector<Scalar, 3> & max_corner = this->mesh_ele_max_corner_vec_[ele_id];

    if (vert_pos[0] < min_corner[0]) return false;
    if (vert_pos[0] > max_corner[0]) return false;
    if (vert_pos[1] < min_corner[1]) return false;
    if (vert_pos[1] > max_corner[1]) return false;
    if (vert_pos[2] < min_corner[2]) return false;
    if (vert_pos[2] > max_corner[2]) return false;

    return true;
}

template <typename Scalar>
bool PDMCollisionMethodSpaceHash<Scalar, 3>::intersectTethedraViaVertexPenetrate(unsigned int ele_id, unsigned int vert_id, const SquareMatrix<Scalar, 3> & A_inverse, const Vector<Scalar, 3> & x0) const
{
    PHYSIKA_ASSERT(driver_);
    VolumetricMesh<Scalar, 3> * mesh = driver_->mesh();
    PHYSIKA_ASSERT(mesh);

    //if element is part of element, then return false
    if (this->isElementContainVertex(ele_id, vert_id) == true)
        return false;
    
    const Vector<Scalar, 3> vert_pos = mesh->vertPos(vert_id);
    if (this->isVertexInsideBoundingVolume(ele_id, vert_pos) == false)
        return false;

    return this->isVertexPenetrateElement(A_inverse, x0, vert_pos);

}

template <typename Scalar>
void PDMCollisionMethodSpaceHash<Scalar, 3>::generateElementFaceNormal(const VolumetricMesh<Scalar, 3> * mesh, std::vector<std::vector<Vector<Scalar, 3> > > & ele_face_normal_vec)
{
    PHYSIKA_ASSERT(mesh);
    ele_face_normal_vec.clear();
    ele_face_normal_vec.resize(mesh->eleNum(), std::vector<Vector<Scalar, 3> >(4, Vector<Scalar, 3>(0.0)));

    //calculate face normal
    #pragma omp parallel for
    for (long long ele_id = 0; ele_id < mesh->eleNum(); ele_id++)
    {
        const std::vector<Vector<Scalar, 3> > & ele_vert_pos = this->mesh_ele_pos_vec_[ele_id];

        Vector<Scalar,3> face_normal(0.0);

        //set face 0
        face_normal = (ele_vert_pos[2] - ele_vert_pos[0]).cross(ele_vert_pos[1] - ele_vert_pos[0]);
        if(face_normal.dot(ele_vert_pos[3] - ele_vert_pos[0]) > 0) //flip normal if not outward
            face_normal *= -1;
        ele_face_normal_vec[ele_id][0] = face_normal;

        //set face 1
        face_normal = (ele_vert_pos[1] - ele_vert_pos[0]).cross(ele_vert_pos[3] - ele_vert_pos[0]);
        if(face_normal.dot(ele_vert_pos[2] - ele_vert_pos[0]) > 0) //flip normal if not outward
            face_normal *= -1;
        ele_face_normal_vec[ele_id][1] = face_normal;

        //set face 2
        face_normal = (ele_vert_pos[3] - ele_vert_pos[0]).cross(ele_vert_pos[2] - ele_vert_pos[0]);
        if(face_normal.dot(ele_vert_pos[1] - ele_vert_pos[0]) > 0) //flip normal if not outward
            face_normal *= -1;
        ele_face_normal_vec[ele_id][2] = face_normal;

        //set face 3
        face_normal = (ele_vert_pos[2] - ele_vert_pos[1]).cross(ele_vert_pos[3] - ele_vert_pos[1]);
        if(face_normal.dot(ele_vert_pos[1] - ele_vert_pos[0]) < 0)
            face_normal *= -1;
        ele_face_normal_vec[ele_id][3] = face_normal;

    }
}

template <typename Scalar>
bool PDMCollisionMethodSpaceHash<Scalar, 3>::isBoundingVolumeOverlap(unsigned int fir_ele_id, unsigned int sec_ele_id)
{
    const Vector<Scalar, 3> & fir_ele_min_corner = this->mesh_ele_min_corner_vec_[fir_ele_id];
    const Vector<Scalar, 3> & fir_ele_max_corner = this->mesh_ele_max_corner_vec_[fir_ele_id];
    const Vector<Scalar, 3> & sec_ele_min_corner = this->mesh_ele_min_corner_vec_[sec_ele_id];
    const Vector<Scalar, 3> & sec_ele_max_corner = this->mesh_ele_max_corner_vec_[sec_ele_id];

    if(fir_ele_min_corner[0] > sec_ele_max_corner[0]) return false;
    if(fir_ele_max_corner[0] < sec_ele_min_corner[0]) return false;
    if(fir_ele_min_corner[1] > sec_ele_max_corner[1]) return false;
    if(fir_ele_max_corner[1] < sec_ele_min_corner[1]) return false;
    if(fir_ele_min_corner[2] > sec_ele_max_corner[2]) return false;
    if(fir_ele_max_corner[2] < sec_ele_min_corner[2]) return false;

    return true;

}

template <typename Scalar>
bool PDMCollisionMethodSpaceHash<Scalar, 3>::intersectTetrahedraViaEdgeIntersect(unsigned int fir_ele_id, unsigned int sec_ele_id, 
                                                                                 const std::vector<Vector<Scalar, 3> > & fir_ele_face_normal, 
                                                                                 const std::vector<Vector<Scalar, 3> > & sec_ele_face_normal,
                                                                                 std::vector<unsigned char> & masks,
                                                                                 std::vector<std::vector<Scalar> > & coord,
                                                                                 std::vector<Vector<Scalar,3> > & teta_to_tetb_vec,
                                                                                 std::vector<Vector<Scalar,3> > & tetb_to_teta_vec)
{
    if (this->isBoundingVolumeOverlap(fir_ele_id, sec_ele_id) == false) 
        return false;

    PHYSIKA_ASSERT(fir_ele_id != sec_ele_id);

    const std::vector<unsigned int> & fir_ele_vert = this->mesh_ele_index_vec_[fir_ele_id];
    const std::vector<unsigned int> & sec_ele_vert = this->mesh_ele_index_vec_[sec_ele_id];
    PHYSIKA_ASSERT(fir_ele_vert.size() == sec_ele_vert.size());
    PHYSIKA_ASSERT(fir_ele_vert.size()==3 || fir_ele_vert.size()==4);

    unsigned int shared_num = 0;
    for (unsigned int i=0; i<fir_ele_vert.size(); i++)
        for (unsigned int j=0; j<sec_ele_vert.size(); j++)
        {
            if (fir_ele_vert[i] == sec_ele_vert[j])
                shared_num++;
        }

    if (shared_num >= 1) return false;

    const std::vector<Vector<Scalar, 3> > & fir_ele_pos = this->mesh_ele_pos_vec_[fir_ele_id];
    const std::vector<Vector<Scalar, 3> > & sec_ele_pos = this->mesh_ele_pos_vec_[sec_ele_id];
    //return GeometryIntersections::intersectTetrahedra(fir_ele_pos, sec_ele_pos);
    return GeometryIntersections::intersectTetrahedra(fir_ele_pos, sec_ele_pos, fir_ele_face_normal, sec_ele_face_normal, masks, coord, teta_to_tetb_vec, tetb_to_teta_vec);
    
}

template <typename Scalar>
Scalar PDMCollisionMethodSpaceHash<Scalar, 3>::tetrahedraOverlapVolume(unsigned int fir_ele_id, unsigned int sec_ele_id)
{
    const std::vector<Vector<Scalar, 3> > & fir_ele_vert_pos = this->mesh_ele_pos_vec_[fir_ele_id];
    const std::vector<Vector<Scalar, 3> > & sec_ele_vert_pos = this->mesh_ele_pos_vec_[sec_ele_id];

    SquareMatrix<Scalar, 3> fir_A(0.0);
    SquareMatrix<Scalar, 3> sec_A(0.0);

    for (unsigned int n = 0; n < 3; n++)
        for (unsigned int m = 0; m < 3; m++)
        {
            fir_A(n, m) = fir_ele_vert_pos[m+1][n] - fir_ele_vert_pos[0][n];
            sec_A(n, m) = sec_ele_vert_pos[m+1][n] - sec_ele_vert_pos[0][n];
        }
    
    SquareMatrix<Scalar, 3> fir_A_inverse = fir_A.inverse();
    SquareMatrix<Scalar, 3> sec_A_inverse = sec_A.inverse();

    Vector<Scalar, 3> fir_min_corner = PDMUtilitySet<Scalar, 3>::computeMinCorner(fir_ele_vert_pos);
    Vector<Scalar, 3> fir_max_corner = PDMUtilitySet<Scalar, 3>::computeMaxCorner(fir_ele_vert_pos);

    //////////////////////////////////////////////////////////////////////////////////////////////

    Vector<Scalar,3> sample_point;
    Vector<Scalar, 3> beta(0.0);

    unsigned int sample_res = 40; //40 sample points in each direction
    Scalar sample_stride = 1.0/(sample_res-1.0);

    unsigned int hit_num = 0;
    unsigned int total_num = 0;

    for(unsigned int idx_x = 0; idx_x < sample_res; ++idx_x)
    {
        beta[0] = idx_x*sample_stride;
        for(unsigned int idx_y = 0; idx_y < sample_res; ++idx_y)
        {
            beta[1] = idx_y*sample_stride;
            if (beta[0] + beta[1] > 1.0) break;

            for(unsigned int idx_z = 0; idx_z < sample_res; ++idx_z)
            {
                beta[2] = idx_z*sample_stride;
                if(beta[0] + beta[1] + beta[2] > 1.0) break;

                total_num ++;
                sample_point = fir_ele_vert_pos[0] + fir_A*beta;

                if(this->isVertexPenetrateElement(sec_A_inverse, sec_ele_vert_pos[0], sample_point) == true)
                    hit_num++;
            }
        }
    }


    PHYSIKA_ASSERT(driver_);
    VolumetricMesh<Scalar, 3> * mesh = driver_->mesh();
    PHYSIKA_ASSERT(mesh);

    Scalar fir_ele_vol = mesh->eleVolume(fir_ele_id);
    
    //std::cout<<"-------------------------------------------------"<<std::endl;
    //std::cout<<"sample points size: "<<total_num<<std::endl;
    //std::cout<<"hit num: "<<hit_num<<std::endl;
    //std::cout<<"fir_ele_vol: "<<fir_ele_vol<<std::endl;
    //std::cout<<"overlap vol: "<<hit_num*1.0/total_num*fir_ele_vol<<std::endl;

    return hit_num*1.0/total_num*fir_ele_vol;

}

//explicit instantiations
template class PDMCollisionMethodSpaceHash<float, 3>;
template class PDMCollisionMethodSpaceHash<double,3>;

}//end of namespace Physika 