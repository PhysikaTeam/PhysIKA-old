/*
 * @file PDM_particle.cpp 
 * @Basic PDParticle class. Particles used for PDM (PeriDynamics Method).
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

#include <limits>
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Dynamics/PDM/PDM_particle.h"

namespace Physika{

template <typename Scalar, int Dim>
PDMParticle<Scalar,Dim>::PDMParticle()
    :Particle(Vector<Scalar,Dim>(0), Vector<Scalar,Dim>(0), 1, 1),init_size_(0),valid_size_(0),delta_(0.0),anisotropic_matrix_(SquareMatrix<Scalar, Dim>::identityMatrix())
{

}

template <typename Scalar, int Dim>
PDMParticle<Scalar, Dim>::PDMParticle(const Vector<Scalar,Dim> &pos, const Vector<Scalar,Dim> &vel, Scalar mass, Scalar vol)
    :Particle(pos, vel, mass, vol),init_size_(0),valid_size_(0),delta_(0.0),anisotropic_matrix_(SquareMatrix<Scalar, Dim>::identityMatrix())
{

}

template <typename Scalar, int Dim>
PDMParticle<Scalar, Dim>::PDMParticle(const Vector<Scalar,Dim> &pos, const Vector<Scalar,Dim> &vel, Scalar mass, Scalar vol, const std::list<PDMFamily<Scalar, Dim> > & family)
    :Particle(pos, vel, mass, vol),family_(family),init_size_(family.size()),valid_size_(family.size()),delta_(0.0),anisotropic_matrix_(SquareMatrix<Scalar, Dim>::identityMatrix())
{

}

template <typename Scalar, int Dim>
PDMParticle<Scalar, Dim>::~PDMParticle()
{

}

template <typename Scalar, int Dim>
PDMParticle<Scalar, Dim> * PDMParticle<Scalar, Dim>::clone() const
{
    return new PDMParticle<Scalar, Dim>(*this);
}

template <typename Scalar, int Dim>
std::list<PDMFamily<Scalar, Dim> > & PDMParticle<Scalar, Dim>::family()
{
    return this->family_;
}

template <typename Scalar, int Dim>
const std::list<PDMFamily<Scalar, Dim> > & PDMParticle<Scalar, Dim>::family() const
{
    return this->family_;
}

template <typename Scalar, int Dim>
void PDMParticle<Scalar, Dim>::setFamily(const std::list<PDMFamily<Scalar, Dim> > & family)
{
    this->family_ = family;
    this->init_size_ = this->family_.size();
    this->valid_size_ = this->family_.size();
}

template <typename Scalar, int Dim>
Scalar PDMParticle<Scalar, Dim>::delta() const
{
    return this->delta_;
}

template <typename Scalar, int Dim>
void PDMParticle<Scalar, Dim>::setDelta(Scalar delta)
{
    this->delta_ = delta;
}

template <typename Scalar, int Dim>
const SquareMatrix<Scalar, Dim> & PDMParticle<Scalar, Dim>::anisotropicMatrix() const
{
    return this->anisotropic_matrix_;
}

template <typename Scalar, int Dim>
void PDMParticle<Scalar, Dim>::setAnistropicMatrix(const SquareMatrix<Scalar, Dim> & anisotropic_matrix)
{
    this->anisotropic_matrix_ = anisotropic_matrix;
}

template <typename Scalar, int Dim>
void PDMParticle<Scalar, Dim>::addFamily(const PDMFamily<Scalar, Dim> & family)
{
    this->family_.push_back(family);
    this->init_size_++;
    this->valid_size_ ++;
}

template <typename Scalar, int Dim>
void PDMParticle<Scalar, Dim>::deleteFamily(typename std::list<PDMFamily<Scalar, Dim> >::iterator & pos_iter)
{
    pos_iter->setCrack(true);
    pos_iter->setVaild(false);
    this->valid_size_ --;

    if (this->direct_neighbor_.count(pos_iter->id()) == 1)
        this->direct_neighbor_.erase(pos_iter->id());

    //need further consideration
    /*
    if (this->direct_neighbor_.size() == 0)
    {
        //delete all its family, if particle has no direct neighbor
        for (std::list<PDMFamily<Scalar, Dim> >::iterator iter = this->family_.begin(); iter!=this->family_.end(); iter++)
        {
            if (iter->isCrack() == false || iter->isVaild() == true)
            {
                iter->setCrack(true);
                iter->setVaild(false);
                this->valid_size_ --;
            }
        }
    }
    */
}

template <typename Scalar, int Dim>
void PDMParticle<Scalar, Dim>::addDirectNeighbor(unsigned int direct_neighbor_id)
{
    this->direct_neighbor_.insert(direct_neighbor_id);
}

template <typename Scalar, int Dim>
void PDMParticle<Scalar, Dim>::addVolume(Scalar vol)
{
    this->vol_ += vol;
}

template <typename Scalar, int Dim>
void PDMParticle<Scalar, Dim>::addVelocity(const Vector<Scalar, Dim> & v)
{
    this->v_ += v;
}

template <typename Scalar, int Dim>
void PDMParticle<Scalar, Dim>::setInitFamilySize(unsigned int init_size)
{
    this->init_size_ = init_size;
}

/*
template <typename Scalar, int Dim>
void PDMParticle<Scalar, Dim>::setValidFamilySize(unsigned int valid_size)
{
    this->valid_size_ = valid_size;
}
*/

template <typename Scalar, int Dim>
unsigned int PDMParticle<Scalar, Dim>::initFamilySize() const
{
    return this->init_size_;
}

template <typename Scalar, int Dim>
unsigned int PDMParticle<Scalar, Dim>::validFamilySize() const
{
    return this->valid_size_;
}

template <typename Scalar, int Dim>
void PDMParticle<Scalar, Dim>::updateFamilyMember()
{
    unsigned int vaild_num = 0;
    for (std::list<PDMFamily<Scalar, Dim> >::iterator iter = this->family_.begin(); iter!=this->family_.end(); iter++)
    {
        //only those uncracked bonds(family members) need to be updated
        if (iter->isCrack() == false && iter->restRelativePosNorm() <= this->delta_)
        {
            iter->setVaild(true);
            vaild_num++;
        }
        else
        {
            iter->setVaild(false);
        }
    }
    this->valid_size_ = vaild_num;
}

template <typename Scalar, int Dim>
const Vector<Scalar, Dim> & PDMParticle<Scalar, Dim>::position() const
{
    return this->x_;
}

template <typename Scalar, int Dim>
const Vector<Scalar, Dim> & PDMParticle<Scalar,Dim>::velocity() const
{
    return this->v_;
}

// explicit instantiation
template class PDMParticle<float,2>;
template class PDMParticle<double,2>;
template class PDMParticle<float,3>;
template class PDMParticle<double,3>;

// Class PDMFamily
template <typename Scalar, int Dim>
PDMFamily<Scalar, Dim>::PDMFamily(const  unsigned int id, const Vector<Scalar, Dim> & rest_relative_pos, const SquareMatrix<Scalar, Dim> & anisotropic_matrix)
    :id_(id), rest_relative_pos_norm_(rest_relative_pos.norm()), 
    unit_rest_relative_pos_(rest_relative_pos/rest_relative_pos_norm_),
    ep_(0.0), ep_limit_(0.0), eb_(0.0), eb_limit_(0.0), ed_(0.0), crack_(false), vaild_(true)
{
    Vector<Scalar, Dim> weight_rest_relative_pos = anisotropic_matrix*rest_relative_pos;
    this->weight_rest_len_ = weight_rest_relative_pos.norm();
}

template <typename Scalar, int Dim>
PDMFamily<Scalar, Dim>::~PDMFamily()
{

}

template <typename Scalar, int Dim>
unsigned int PDMFamily<Scalar, Dim>::id() const
{
    return this->id_;
}

template <typename Scalar, int Dim>
Scalar PDMFamily<Scalar, Dim>::restRelativePosNorm() const
{
    return this->rest_relative_pos_norm_;
}

template <typename Scalar, int Dim>
const Vector<Scalar, Dim> & PDMFamily<Scalar, Dim>::unitRestRelativePos() const
{
    return this->unit_rest_relative_pos_;
}

template <typename Scalar, int Dim>
Scalar PDMFamily<Scalar, Dim>::curRelativePosNorm() const
{
    return this->cur_relative_pos_norm_;
}

template <typename Scalar, int Dim>
const Vector<Scalar, Dim> & PDMFamily<Scalar, Dim>::unitCurRelativePos() const
{
    return this->unit_cur_relative_pos_;
}

template <typename Scalar, int Dim>
bool PDMFamily<Scalar, Dim>::isVaild() const
{
    return this->vaild_;
}

template <typename Scalar, int Dim>
bool PDMFamily<Scalar, Dim>::isCrack() const
{
    return this->crack_;
}

template <typename Scalar, int Dim>
Scalar PDMFamily<Scalar, Dim>::weightRestLen() const
{
    return this->weight_rest_len_;
}

template <typename Scalar, int Dim>
Scalar PDMFamily<Scalar, Dim>::ep() const
{
    return this->ep_;
}

template <typename Scalar, int Dim>
Scalar PDMFamily<Scalar, Dim>::epStretchLimit() const
{
    return this->ep_limit_/this->rest_relative_pos_norm_;
}

template <typename Scalar, int Dim>
Scalar PDMFamily<Scalar, Dim>::epLimit() const
{
    return this->ep_limit_;
}

template <typename Scalar, int Dim>
Scalar PDMFamily<Scalar, Dim>::eb() const
{
    return this->eb_;
}

template <typename Scalar, int Dim>
Scalar PDMFamily<Scalar, Dim>::ebStretchLimit() const
{
    return this->eb_limit_/this->rest_relative_pos_norm_;
}

template <typename Scalar, int Dim>
Scalar PDMFamily<Scalar, Dim>::ebLimit() const
{
    return this->eb_limit_;
}

template <typename Scalar, int Dim>
Scalar PDMFamily<Scalar, Dim>::ed() const
{
    return this->ed_;
}

template <typename Scalar, int Dim>
void PDMFamily<Scalar, Dim>::setCurRelativePos(const Vector<Scalar, Dim> & cur_relative_pos)
{
    Scalar cur_relative_norm = cur_relative_pos.norm();
    if (cur_relative_norm >= 1.5e-7)  // a least value of 1.5e-7 is required
    {
        this->cur_relative_pos_norm_ = cur_relative_norm;
        this->unit_cur_relative_pos_ = cur_relative_pos/this->cur_relative_pos_norm_;
    }
    else
    {
        this->cur_relative_pos_norm_ = 1.5e-7;
    }
}

template <typename Scalar, int Dim>
void PDMFamily<Scalar, Dim>::setVaild(bool vaild)
{
    this->vaild_ = vaild;
}

template <typename Scalar, int Dim>
void PDMFamily<Scalar, Dim>::setCrack(bool crack)
{
    this->crack_ = crack;
}

template <typename Scalar, int Dim>
void PDMFamily<Scalar, Dim>::setEpStretchLimit(Scalar ep_stretch_limit)
{
    if (ep_stretch_limit < 0.0)
    {
        std::cerr<<"error: ep_limit must be equal or greater than zero!\n";
        std::exit(EXIT_FAILURE);
    }
    this->ep_limit_ = ep_stretch_limit*this->rest_relative_pos_norm_;
}

template <typename Scalar, int Dim>
void PDMFamily<Scalar, Dim>::setEp(Scalar ep)
{
    this->ep_ = ep;
}

template <typename Scalar, int Dim>
void PDMFamily<Scalar, Dim>::addEp(Scalar delta_ep)
{
    this->ep_ += delta_ep;
}

template <typename Scalar, int Dim>
void PDMFamily<Scalar, Dim>::setEbStretchLimit(Scalar eb_stretch_limit)
{
    if (eb_stretch_limit < 0.0)
    {
        std::cerr<<"error: eb_limit must be equal or greater than zero!\n";
        std::exit(EXIT_FAILURE);
    }
    this->eb_limit_ = eb_stretch_limit*this->rest_relative_pos_norm_;
}

template <typename Scalar, int Dim>
void PDMFamily<Scalar, Dim>::setEb(Scalar eb)
{
    this->eb_ = eb;
}

template <typename Scalar, int Dim>
void PDMFamily<Scalar, Dim>::addEb(Scalar delta_eb)
{
    this->eb_ += delta_eb;
}

template <typename Scalar, int Dim>
void PDMFamily<Scalar, Dim>::setEd(Scalar ed)
{
    this->ed_ = ed;
}



// explicit instantiation
template class PDMFamily<float,2>;
template class PDMFamily<double,2>;
template class PDMFamily<float,3>;
template class PDMFamily<double,3>;


}// end of namespace Physika