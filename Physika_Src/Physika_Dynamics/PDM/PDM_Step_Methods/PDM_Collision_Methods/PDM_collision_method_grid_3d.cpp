/*
 * @file PDM_collision_method_grid_3d.cpp 
 * @brief class of collision method(three dim) for PDM drivers.
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

#include "Physika_Core/Timer/timer.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Dynamics/PDM/PDM_particle.h"
#include "Physika_Dynamics/PDM/PDM_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Collision_Methods/PDM_collision_method_grid_3d.h"

namespace Physika{

template<typename Scalar>
PDMCollisionMethodGrid<Scalar,3>::PDMCollisionMethodGrid()
    :lambda_(0),x_spacing_(0),y_spacing_(0),z_spacing_(0),
    x_bin_num_(0),y_bin_num_(0),z_bin_num_(0),
    bin_start_point_(0),space_hash_bin_(0)
{

}

template<typename Scalar>
PDMCollisionMethodGrid<Scalar,3>::~PDMCollisionMethodGrid()
{
    if (space_hash_bin_)
    {
        delete [] space_hash_bin_;
    }
}

template <typename Scalar>
Scalar PDMCollisionMethodGrid<Scalar,3>::lambda() const
{
    return this->lambda_;
}

template <typename Scalar>
void PDMCollisionMethodGrid<Scalar,3>::setLambda(Scalar lambda)
{
    this->lambda_ = lambda;
}

template <typename Scalar>
Vector<Scalar,3> PDMCollisionMethodGrid<Scalar,3>::binStartPoint() const
{
    return this->bin_start_point_;
}

template <typename Scalar>
void PDMCollisionMethodGrid<Scalar,3>::setBinStartPoint(Vector<Scalar,3> bin_start_point)
{
    this->bin_start_point_ = bin_start_point;
}

template <typename Scalar>
Scalar PDMCollisionMethodGrid<Scalar,3>::XSpacing() const
{
    return this->x_spacing_;
}

template <typename Scalar>
Scalar PDMCollisionMethodGrid<Scalar,3>::YSpacing() const
{
    return this->y_spacing_;
}

template <typename Scalar>
Scalar PDMCollisionMethodGrid<Scalar,3>::ZSpacing() const
{
    return this->z_spacing_;
}

template <typename Scalar>
void PDMCollisionMethodGrid<Scalar,3>::setXYZSpacing(Scalar x_spacing, Scalar y_spacing, Scalar z_spacing)
{
    PHYSIKA_ASSERT(x_spacing>0.0 && y_spacing>0.0 && z_spacing>0.0);
    this->x_spacing_ = x_spacing;
    this->y_spacing_ = y_spacing;
    this->z_spacing_ = z_spacing;
}

template <typename Scalar>
void PDMCollisionMethodGrid<Scalar,3>::setUnifySpacing(Scalar spacing)
{
    PHYSIKA_ASSERT(spacing>0.0);
    this->x_spacing_ = spacing;
    this->y_spacing_ = spacing;
    this->z_spacing_ = spacing;
}

template <typename Scalar>
unsigned int PDMCollisionMethodGrid<Scalar,3>::XBinNum() const
{
    return this->x_bin_num_;
}

template <typename Scalar>
unsigned int PDMCollisionMethodGrid<Scalar,3>::YBinNum() const
{
    return this->y_bin_num_;
}

template <typename Scalar>
unsigned int PDMCollisionMethodGrid<Scalar,3>::ZBinNum() const
{
    return this->z_bin_num_;
}

template <typename Scalar>
void PDMCollisionMethodGrid<Scalar,3>::setXYZBinNum(unsigned int x_bin_num, unsigned int y_bin_num, unsigned int z_bin_num)
{
    this->x_bin_num_ = x_bin_num;
    this->y_bin_num_ = y_bin_num;
    this->z_bin_num_ = z_bin_num;
    if (space_hash_bin_)
    {
        delete [] space_hash_bin_;
    }
    this->space_hash_bin_ = new std::vector<unsigned int>[x_bin_num_*y_bin_num_*z_bin_num_];
    PHYSIKA_ASSERT(this->space_hash_bin_);
}

template <typename Scalar>
void PDMCollisionMethodGrid<Scalar,3>::setUnifyBinNum(unsigned int bin_num)
{
    this->x_bin_num_ = bin_num;
    this->y_bin_num_ = bin_num;
    this->z_bin_num_ = bin_num;
    if (space_hash_bin_)
    {
        delete [] space_hash_bin_;
    }
    this->space_hash_bin_ = new std::vector<unsigned int>[x_bin_num_*y_bin_num_*z_bin_num_];
    PHYSIKA_ASSERT(this->space_hash_bin_);
}


template <typename Scalar>
void PDMCollisionMethodGrid<Scalar,3>::resetHashBin()
{
    unsigned int num_bins = x_bin_num_*y_bin_num_*z_bin_num_;
    //#pragma omp parallel for
    for (unsigned int bin_idx =0; bin_idx<num_bins; bin_idx++)
    {
        space_hash_bin_[bin_idx].clear();
    }
}

template <typename Scalar>
void PDMCollisionMethodGrid<Scalar,3>::locateParticleBin()
{
    PHYSIKA_ASSERT(driver_);
    unsigned int num_particles = driver_->numSimParticles();
    for (unsigned int par_idx =0; par_idx<num_particles; par_idx++)
    {
        Vector<Scalar, 3> par_pos = driver_->particleCurrentPosition(par_idx);
        par_pos -= bin_start_point_;
        long bin_x_pos = par_pos[0]/x_spacing_;
        long bin_y_pos = par_pos[1]/y_spacing_;
        long bin_z_pos = par_pos[2]/z_spacing_;

        if(isOutOfRange(bin_x_pos, bin_y_pos, bin_z_pos) == false)
        {
            unsigned int bin_pos = getHashBinPos(bin_x_pos, bin_y_pos, bin_z_pos);
            space_hash_bin_[bin_pos].push_back(par_idx);
        }
    }
}

template <typename Scalar>
void PDMCollisionMethodGrid<Scalar,3>::collisionDectectionAndResponse()
{
    PHYSIKA_ASSERT(driver_);

    unsigned int collision_num = 0;
    //#pragma omp parallel for
    for (unsigned int x_bin_idx = 0; x_bin_idx<x_bin_num_; x_bin_idx++)
    for (unsigned int y_bin_idx = 0; y_bin_idx<y_bin_num_; y_bin_idx++)
    for (unsigned int z_bin_idx = 0; z_bin_idx<z_bin_num_; z_bin_idx++)
    {
        unsigned int bin_idx = getHashBinPos(x_bin_idx, y_bin_idx, z_bin_idx);
        unsigned int par_num = space_hash_bin_[bin_idx].size();

        for (unsigned int par_idx =0; par_idx < par_num; par_idx++)
        {
            unsigned int par_id = space_hash_bin_[bin_idx][par_idx];
            Vector<Scalar,3> par_pos = driver_->particleCurrentPosition(par_id);
            for (int i=-1; i<=1; i++)
            for (int j=-1; j<=1; j++)
            for (int k=-1; k<=1; k++)
            {
                long test_x_bin_idx = x_bin_idx+i;
                long test_y_bin_idx = y_bin_idx+j;
                long test_z_bin_idx = z_bin_idx+k;

                if (isOutOfRange(test_x_bin_idx, test_y_bin_idx, test_z_bin_idx) == false)
                {
                    unsigned int test_bin_idx = getHashBinPos(test_x_bin_idx, test_y_bin_idx, test_z_bin_idx);
                    unsigned int test_par_num = space_hash_bin_[test_bin_idx].size();

                    for (unsigned int test_par_idx = 0; test_par_idx<test_par_num; test_par_idx++)
                    {
                        unsigned int test_par_id = space_hash_bin_[test_bin_idx][test_par_idx];
                                        
                        if (par_id >= test_par_id) // "=" to exclude the particle itself
                        {
                            continue;
                        }

                        Vector<Scalar,3> test_par_pos = driver_->particleCurrentPosition(test_par_id);
                        Vector<Scalar,3> relative_pos = par_pos - test_par_pos;
                        if (relative_pos.normSquared() < lambda_*lambda_/4)
                        {
                            collision_num++;
                            //std::cout<<"collision detection: "<<par_id<<" with "<< test_par_id<<std::endl;
                            Scalar relative_pos_norm = relative_pos.norm();
                            Scalar distance = relative_pos_norm - lambda_/2;
                            Vector<Scalar,3> force = Kc_*distance*distance/relative_pos_norm*relative_pos;
                            // add force
                            driver_->addParticleForce(par_id, force);
                            driver_->addParticleForce(test_par_id, -force);
                        }

                    }
                }

            }
        }

    }
    std::cout<<"collision num: "<<collision_num<<std::endl;
}

template <typename Scalar>
void PDMCollisionMethodGrid<Scalar,3>::collisionMethod()
{
    Timer timer;

    timer.startTimer();
    resetHashBin();
    timer.stopTimer();
    std::cout<<"reset time: "<<timer.getElapsedTime()<<std::endl;

    timer.startTimer();
    locateParticleBin();
    timer.stopTimer();
    std::cout<<"locate time: "<<timer.getElapsedTime()<<std::endl;

    timer.startTimer();
    collisionDectectionAndResponse();
    timer.stopTimer();
    std::cout<<"detection and response time: "<<timer.getElapsedTime()<<std::endl;
}

template <typename Scalar>
void PDMCollisionMethodGrid<Scalar,3>::initParticleFamily(Scalar max_delta)
{
    PHYSIKA_ASSERT(driver_);

    resetHashBin();
    locateParticleBin();

    unsigned int total_family_name = 0;

    #pragma omp parallel for
    for (long long    x_bin_idx = 0; x_bin_idx<x_bin_num_; x_bin_idx++)
    for (unsigned int y_bin_idx = 0; y_bin_idx<y_bin_num_; y_bin_idx++)
    for (unsigned int z_bin_idx = 0; z_bin_idx<z_bin_num_; z_bin_idx++)
    {
        unsigned int bin_idx = getHashBinPos(x_bin_idx, y_bin_idx, z_bin_idx);
        unsigned int par_num = space_hash_bin_[bin_idx].size();

        for (unsigned int par_idx =0; par_idx < par_num; par_idx++)
        {
            unsigned int par_id = space_hash_bin_[bin_idx][par_idx];
            Vector<Scalar,3> par_pos = driver_->particleCurrentPosition(par_id);
            for (int i=-1; i<=1; i++)
            for (int j=-1; j<=1; j++)
            for (int k=-1; k<=1; k++)
            {
                long test_x_bin_idx = x_bin_idx+i;
                long test_y_bin_idx = y_bin_idx+j;
                long test_z_bin_idx = z_bin_idx+k;

                if (isOutOfRange(test_x_bin_idx, test_y_bin_idx, test_z_bin_idx) == false)
                {
                    unsigned int test_bin_idx = getHashBinPos(test_x_bin_idx, test_y_bin_idx, test_z_bin_idx);
                    unsigned int test_par_num = space_hash_bin_[test_bin_idx].size();

                    for (unsigned int test_par_idx = 0; test_par_idx<test_par_num; test_par_idx++)
                    {
                        unsigned int test_par_id = space_hash_bin_[test_bin_idx][test_par_idx];
                        if (par_id != test_par_id)
                        {
                            Scalar delta_squared = max_delta*max_delta;
                            if ((driver_->particleRestPosition(par_id)-driver_->particleRestPosition(test_par_id)).normSquared()<delta_squared)
                            {
                                Vector<Scalar, 3> rest_relative_pos = driver_->particleRestPosition(test_par_id) - driver_->particleRestPosition(par_id);
                                Vector<Scalar, 3> cur_relative_pos = driver_->particleCurrentPosition(test_par_id) - driver_->particleCurrentPosition(par_id);

                                PDMParticle<Scalar,3> & particle = driver_->particle(par_id);
                                const SquareMatrix<Scalar, 3> & anisotropic_matrix = particle.anisotropicMatrix();

                                PDMFamily<Scalar, 3> family(test_par_id, rest_relative_pos, anisotropic_matrix);
                                family.setCurRelativePos(cur_relative_pos);
                                particle.addFamily(family);

                                #pragma omp atomic
                                total_family_name ++;
                            }
                        }

                    }
                }

            }
        }

    }

    std::cout<<"total family num: "<<total_family_name<<std::endl;
    std::cout<<"memory per family:             "<<sizeof(PDMFamily<Scalar, 3>)<<"Byte"<<std::endl;
    std::cout<<"memory cost of all family:     "<<total_family_name*sizeof(PDMFamily<Scalar, 3>)/(1024*1024)<<"MB"<<std::endl;
    std::system("pause");

}


template <typename Scalar>
unsigned int PDMCollisionMethodGrid<Scalar,3>::getHashBinPos(unsigned int x_pos, unsigned int y_pos, unsigned int z_pos)
{
    return z_pos*(x_bin_num_*y_bin_num_)+y_pos*x_bin_num_+x_pos;
}

template <typename Scalar>
bool PDMCollisionMethodGrid<Scalar,3>::isOutOfRange(long x_pos, long y_pos, long z_pos)
{
    if ( x_pos >= x_bin_num_ || x_pos <0 || y_pos >= y_bin_num_ || y_pos <0 || z_pos >= z_bin_num_ || z_pos <0 )
    {
        return true;
    }
    return false;
}

//explicit instantiations
template class PDMCollisionMethodGrid<float,3>;
template class PDMCollisionMethodGrid<double,3>;

}// end of namespace Physika