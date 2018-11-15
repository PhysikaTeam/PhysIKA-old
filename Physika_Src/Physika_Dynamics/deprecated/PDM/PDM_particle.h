/*
 * @file PDM_particle.h 
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

# ifndef PHYSIKA_DYNAMICS_PDM_PDM_PARTICLE_H
# define PHYSIKA_DYNAMICS_PDM_PDM_PARTICLE_H

#include <list>
#include <set>
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Dynamics/Particles/particle.h"

namespace Physika{

template <typename Scalar, int Dim> class PDMFamily;

template <typename Scalar, int Dim>
class PDMParticle: public Particle<Scalar, Dim>
{
public:
    // constructor and destructor
    PDMParticle(); // default mass and vol is 1 rather than 0, to avoid meaningless data
    PDMParticle(const Vector<Scalar,Dim> &pos, const Vector<Scalar,Dim> &vel, Scalar mass, Scalar vol);
    PDMParticle(const Vector<Scalar,Dim> &pos, const Vector<Scalar,Dim> &vel, Scalar mass, Scalar vol, const std::list<PDMFamily<Scalar, Dim> > & family);
    virtual ~PDMParticle();

    virtual PDMParticle<Scalar,Dim>* clone() const;

    // getter and setter
    std::list<PDMFamily<Scalar, Dim> > &       family();
    const std::list<PDMFamily<Scalar, Dim> > & family() const;
    void   setFamily(const std::list<PDMFamily<Scalar, Dim> > & family);
    Scalar delta() const;
    void   setDelta(Scalar delta);
    const SquareMatrix<Scalar, Dim> & anisotropicMatrix() const;
    void setAnistropicMatrix(const SquareMatrix<Scalar, Dim> & anisotropic_matrix);

    void addFamily(const PDMFamily<Scalar, Dim> & family);
    // note: we do not provide a deleter according to a relative position in list due to its low efficiency 
    void deleteFamily(typename std::list<PDMFamily<Scalar, Dim> >::iterator & pos_iter );

    void addDirectNeighbor(unsigned int direct_neighbor_id);

    void addVolume(Scalar vol);
    void addVelocity(const Vector<Scalar, Dim> & v);

    void setInitFamilySize(unsigned int init_size);
    //void setValidFamilySize(unsigned int valid_size);
    unsigned int initFamilySize() const;
    unsigned int validFamilySize() const;

    // update family member
    void updateFamilyMember();

    // override position() and velocity() to imporve performance
    const Vector<Scalar, Dim> & position() const;
    const Vector<Scalar, Dim> & velocity() const;

protected:
    // the family of material point specified by horizon delta, note that the user should guarantee every idx is unique in family,
    // here idx is the global id in PDM Driver.
    std::list<PDMFamily<Scalar, Dim> > family_;
    unsigned int init_size_;
    unsigned int valid_size_;

    std::set<unsigned int> direct_neighbor_;

    // delta: horizon
    Scalar delta_; //default:0.0, should be specified through driver

    // anisotropic matrix, need further consideration
    // currently, only stated-based PDM support anisotropy
    SquareMatrix<Scalar, Dim> anisotropic_matrix_; //default: identity matrix
};

template <typename Scalar, int Dim>
class PDMFamily
{
public:
    // default constructor not provided 
    PDMFamily(const unsigned int id, const Vector<Scalar, Dim> & rest_relative_pos, const SquareMatrix<Scalar, Dim> & anisotropic_matrix);
    virtual ~PDMFamily();

    //getter
    unsigned int id() const;

    Scalar restRelativePosNorm() const;
    const Vector<Scalar, Dim> & unitRestRelativePos() const;

    Scalar curRelativePosNorm() const;
    const Vector<Scalar, Dim> & unitCurRelativePos() const;
    bool isVaild() const;
    bool isCrack() const;

    Scalar weightRestLen() const;

    //plasticity
    Scalar ep() const;
    Scalar epStretchLimit() const;
    Scalar epLimit() const;

    Scalar eb() const;
    Scalar ebStretchLimit() const;
    Scalar ebLimit() const;

    //setter
    void setCurRelativePos(const Vector<Scalar, Dim> & cur_relative_pos);
    void setVaild(bool vaild);
    void setCrack(bool crack);

    void addEp(Scalar delta_ep);
    void setEp(Scalar ep);
    void setEpStretchLimit(Scalar ep_limit);

    void addEb(Scalar delta_eb);
    void setEb(Scalar eb);
    void setEbStretchLimit(Scalar eb_limit);
    
    //for visco plasticity
    Scalar ed() const;
    void setEd(Scalar last_ed);

protected:
    const unsigned int id_;

    const Scalar rest_relative_pos_norm_;              // norm of rest relative position, should not be modified since its initialization
    const Vector<Scalar, Dim> unit_rest_relative_pos_; // unit of rest relative position, should not be modified since its initialization
    
    Scalar cur_relative_pos_norm_;                  // norm of current relative position, modified along with cur_relative_pos_
    Vector<Scalar, Dim> unit_cur_relative_pos_;     // unit vector of current relative position, modified along with cur_relative_pos_

    Scalar weight_rest_len_;        //weight length by anisotopy

    Scalar ep_;                     // plasticity part of extension, initial value: 0.0
    Scalar ep_limit_;               // the limit of ep in absolute value manner, must be no less than 0.0, default: 0.0

    Scalar eb_;                     // back extension, defalut: 0.0
    Scalar eb_limit_;               // limit of back extension
     
    Scalar ed_;                     // used only for visco plasticity, default: 0.0

    bool vaild_; // whether the family is within the horizon, default: false
    bool crack_; // whether the family is crack, default: true

};

} // end of namespace Physika

#endif  // PHYSIKA_DYNAMICS_PDM_PDM_PARTICLE_H