/*
 * @file PDM_collision_method_mesh_2d.cpp
 * @brief class of collision method(two dim) based on mesh for PDM drivers.
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

#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"
#include "Physika_Geometry/Geometry_Intersections/tetrahedron_tetrahedron_intersection.h"

#include "Physika_Dynamics/PDM/PDM_particle.h"
#include "Physika_Dynamics/PDM/PDM_base.h"
#include "Physika_Dynamics/PDM/PDM_Utility_Set/PDM_utility_set.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Collision_Methods/PDM_collision_method_mesh_3d.h"

namespace Physika{

template <typename Scalar>
PDMCollisionMethodMesh<Scalar, 3>::PDMCollisionMethodMesh()
{

}

template <typename Scalar>
PDMCollisionMethodMesh<Scalar, 3>::~PDMCollisionMethodMesh()
{
    
}

template <typename Scalar>
void PDMCollisionMethodMesh<Scalar, 3>::setDriver(PDMBase<Scalar,3> * driver)
{
    this->PDMCollisionMethodBase::setDriver(driver);

    VolumetricMesh<Scalar, 3> * mesh = driver_->mesh();
    PHYSIKA_ASSERT(mesh);

    unsigned int num_particles = driver_->numSimParticles(); 
    PHYSIKA_ASSERT(mesh->eleNum() == num_particles);

    this->mesh_ele_index_vec_.resize(num_particles, std::vector<unsigned int> ());
    this->mesh_ele_pos_vec_.resize(num_particles, std::vector<Vector<Scalar, 3> >());

}

template <typename Scalar>
void PDMCollisionMethodMesh<Scalar, 3>::locateParticleBin()
{
    PHYSIKA_ASSERT(driver_);
    unsigned int num_particles = driver_->numSimParticles();
    VolumetricMesh<Scalar, 3> * mesh = driver_->mesh();
    PHYSIKA_ASSERT(mesh);

    unsigned int max_bin_num = 0;
    for (unsigned int par_idx =0; par_idx<num_particles; par_idx++)
    {
        std::vector<Vector<Scalar, 3> > ele_pos_vec;
        mesh->eleVertPos(par_idx, ele_pos_vec);

        Vector<Scalar, 3> min_corner = PDMUtilitySet<Scalar, 3>::computeMinCorner(ele_pos_vec);
        Vector<Scalar, 3> max_corner = PDMUtilitySet<Scalar, 3>::computeMaxCorner(ele_pos_vec);

        min_corner -= this->bin_start_point_;
        max_corner -= this->bin_start_point_;

        //std::cout<<"min_corner: "<<min_corner<<std::endl;
        //std::cout<<"max_corner: "<<max_corner<<std::endl;


        long bin_x_pos_min = min_corner[0]/this->x_spacing_;
        long bin_y_pos_min = min_corner[1]/this->y_spacing_;
        long bin_z_pos_min = min_corner[2]/this->z_spacing_;

        long bin_x_pos_max = max_corner[0]/this->x_spacing_;
        long bin_y_pos_max = max_corner[1]/this->y_spacing_;
        long bin_z_pos_max = max_corner[2]/this->z_spacing_;

        //std::cout<<"bin_x_pos_min: "<<bin_x_pos_min<<std::endl;
        //std::cout<<"bin_x_pos_max: "<<bin_x_pos_max<<std::endl;

        unsigned int bin_num = (bin_x_pos_max - bin_x_pos_min + 1)*(bin_y_pos_max - bin_y_pos_min + 1)*(bin_z_pos_max - bin_z_pos_min + 1);
        max_bin_num = max(max_bin_num, bin_num);

        /*
        if (bin_num == 8)
        {
            std::cout<<"par_idx: "<<par_idx<<std::endl;
            std::cout<<"min_corner: "<<min_corner<<std::endl;
            std::cout<<"max_corner: "<<max_corner<<std::endl;
            std::cout<<"bin_x_pos_min: "<<bin_x_pos_min<<std::endl;
            std::cout<<"bin_x_pos_max: "<<bin_x_pos_max<<std::endl;
            std::cout<<"bin_y_pos_min: "<<bin_y_pos_min<<std::endl;
            std::cout<<"bin_y_pos_max: "<<bin_y_pos_max<<std::endl;
            std::cout<<"bin_z_pos_min: "<<bin_z_pos_min<<std::endl;
            std::cout<<"bin_z_pos_max: "<<bin_z_pos_max<<std::endl;
            std::system("pause");
        }
        */

        for (long x_bin_idx = bin_x_pos_min; x_bin_idx <= bin_x_pos_max; x_bin_idx++)
        for (long y_bin_idx = bin_y_pos_min; y_bin_idx <= bin_y_pos_max; y_bin_idx++)
        for (long z_bin_idx = bin_z_pos_min; z_bin_idx <= bin_z_pos_max; z_bin_idx++)
        {
            if (this->isOutOfRange(x_bin_idx, y_bin_idx, z_bin_idx) == false)
            {
                unsigned int bin_pos = getHashBinPos(x_bin_idx, y_bin_idx, z_bin_idx);
                space_hash_bin_[bin_pos].push_back(par_idx);
            }
        }

    }
    std::cout<<"max_bin_num: "<<max_bin_num<<std::endl;
}

template <typename Scalar>
void PDMCollisionMethodMesh<Scalar, 3>::collisionDectectionAndResponse()
{
    //refresh pre stored mesh information including element vert index and position
    this->refreshPreStoredMeshInfomation();

    PHYSIKA_ASSERT(driver_);
    VolumetricMesh<Scalar, 3> * mesh = driver_->mesh();
    PHYSIKA_ASSERT(mesh);

    std::set<std::pair<unsigned int, unsigned int> > collision_result;

    //collision detection
    for (unsigned int z_bin_idx = 0; z_bin_idx<z_bin_num_; z_bin_idx++)
    for (unsigned int y_bin_idx = 0; y_bin_idx<y_bin_num_; y_bin_idx++)
    for (unsigned int x_bin_idx = 0; x_bin_idx<x_bin_num_; x_bin_idx++)
    {
        unsigned int bin_idx = getHashBinPos(x_bin_idx, y_bin_idx, z_bin_idx);
        unsigned int par_num = space_hash_bin_[bin_idx].size();

        //std::cout<<"bin_idx & par_num: "<<bin_idx<<" "<<par_num<<std::endl;

        for (unsigned int i=0; i<par_num; i++)
        for (unsigned int j=i+1; j<par_num; j++)
        {
            unsigned int par_i = space_hash_bin_[bin_idx][i];
            unsigned int par_j = space_hash_bin_[bin_idx][j];
                
            std::pair<unsigned int, unsigned int> collision_pair(par_i, par_j);
            if (collision_pair.first > collision_pair.second) std::swap(collision_pair.first, collision_pair.second);

            if (collision_result.count(collision_pair) == 1) continue;

            if (this->intersectTetrahedra(par_i, par_j) == true)
            {
                collision_result.insert(collision_pair);
                //std::cout<<"collision i & j: "<<par_i<<" "<<par_j<<std::endl;
                //std::system("pause");
            }
                                
        }
        
    }

    //collision response
    for (std::set<std::pair<unsigned int, unsigned int> >::iterator iter = collision_result.begin(); iter != collision_result.end(); iter++)
    {
        unsigned int par_i = iter->first;
        unsigned int par_j = iter->second;

        const Vector<Scalar, 3> relative_pos = driver_->particleCurrentPosition(par_i) - driver_->particleCurrentPosition(par_j);
        Scalar relative_pos_norm = relative_pos.norm();

        Vector<Scalar, 3> force = (Kc_/relative_pos_norm)*relative_pos;

        // add force
        driver_->addParticleForce(par_i, force);
        driver_->addParticleForce(par_j, -force);
    }

    std::cout<<"**********************************************\n";
    std::cout<<"collision num: "<<collision_result.size()<<std::endl;
    std::cout<<"**********************************************\n";
}

template <typename Scalar>
void PDMCollisionMethodMesh<Scalar, 3>::refreshPreStoredMeshInfomation()
{
    VolumetricMesh<Scalar, 3> * mesh = driver_->mesh();
    PHYSIKA_ASSERT(mesh);

    unsigned int num_particles = driver_->numSimParticles(); 
    PHYSIKA_ASSERT(mesh->eleNum() == num_particles);

    for (unsigned int par_id = 0; par_id < num_particles; par_id++)
    {
        //element vert index
        std::vector<unsigned int> ele_vert;
        mesh->eleVertIndex(par_id, ele_vert);
        this->mesh_ele_index_vec_[par_id] = ele_vert;

        //element vert pos
        std::vector<Vector<Scalar, 3> > ele_pos_vec;
        mesh->eleVertPos(par_id, ele_pos_vec);
        this->mesh_ele_pos_vec_[par_id] = ele_pos_vec;
    }
}

template <typename Scalar>
bool PDMCollisionMethodMesh<Scalar, 3>::intersectTetrahedra(unsigned int fir_ele_id, unsigned int sec_ele_id) const
{
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

    const std::vector<Vector<Scalar, 3> > & fir_ele_pos_vec = this->mesh_ele_pos_vec_[fir_ele_id];
    const std::vector<Vector<Scalar, 3> > & sec_ele_pos_vec = this->mesh_ele_pos_vec_[sec_ele_id];

    PHYSIKA_ASSERT(fir_ele_pos_vec.size() == sec_ele_pos_vec.size());
    PHYSIKA_ASSERT(fir_ele_pos_vec.size() == 4);
    return GeometryIntersections::intersectTetrahedra(fir_ele_pos_vec, sec_ele_pos_vec);
}

//explicit instantiations
template class PDMCollisionMethodMesh<float, 3>;
template class PDMCollisionMethodMesh<double,3>;

}//end of namespace Physika