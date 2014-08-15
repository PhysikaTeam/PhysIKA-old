/*
 * @file CPDI_update_method.cpp 
 * @Brief the particle domain update procedure introduced in paper:
 *        "A convected particle domain interpolation technique to extend applicability of
 *         the material point method for problems involving massive deformations"
 *        It's the base class of all update methods derived from CPDI
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <cstdlib>
#include <iostream>
#include "Physika_Core/Arrays/array_Nd.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Dynamics/Particles/solid_particle.h"
#include "Physika_Dynamics/MPM/CPDI_mpm_solid.h"
#include "Physika_Dynamics/MPM/CPDI_Update_Methods/CPDI_update_method.h"

namespace Physika{

template <typename Scalar, int Dim>
CPDIUpdateMethod<Scalar,Dim>::CPDIUpdateMethod()
    :cpdi_driver_(NULL)
{
}

template <typename Scalar, int Dim>
CPDIUpdateMethod<Scalar,Dim>::~CPDIUpdateMethod()
{
}

template <typename Scalar, int Dim>
void CPDIUpdateMethod<Scalar,Dim>::updateParticleDomain()
{
    PHYSIKA_STATIC_ASSERT(Dim==2||Dim==3,"Invalid dimension specified!");
    PHYSIKA_ASSERT(this->cpdi_driver_);
    ArrayND<Vector<Scalar,Dim>,Dim> particle_domain;
    for(unsigned int i = 0; i < cpdi_driver_->particleNum(); ++i)
    {
        SquareMatrix<Scalar,Dim> deform_grad = (cpdi_driver_->particle(i)).deformationGradient();
        cpdi_driver_->initialParticleDomain(i,particle_domain);
        Vector<Scalar,Dim> particle_pos = (cpdi_driver_->particle(i)).position();
        if(Dim==2)
        {
            Vector<unsigned int,Dim> corner_idx(0);
            Vector<Scalar,Dim> min_corner = particle_domain(corner_idx);
            corner_idx[0] = 1;
            Vector<Scalar,Dim> x_corner = particle_domain(corner_idx);
            corner_idx[0] = 0; corner_idx[1] = 1;
            Vector<Scalar,Dim> y_corner = particle_domain(corner_idx);
            Vector<Scalar,Dim> r_x = x_corner - min_corner;
            Vector<Scalar,Dim> r_y = y_corner - min_corner;
            //update parallelogram
            r_x = deform_grad * r_x;
            r_y = deform_grad * r_y;
            //update 4 corners
            min_corner = particle_pos - 0.5*r_x - 0.5*r_y;
            for(unsigned int idx_x = 0; idx_x < 2; ++idx_x)
                for(unsigned int idx_y = 0; idx_y < 2; ++idx_y)
                {
                    corner_idx[0] = idx_x;
                    corner_idx[1] = idx_y;
                    particle_domain(corner_idx) = min_corner + idx_x*r_x + idx_y*r_y;
                }
            cpdi_driver_->setCurrentParticleDomain(i,particle_domain);
        }
        if(Dim==3)
        {
//TO DO
        }
    }
}

template <typename Scalar, int Dim>
void CPDIUpdateMethod<Scalar,Dim>::setCPDIDriver(CPDIMPMSolid<Scalar,Dim> *cpdi_driver)
{
    if(cpdi_driver==NULL)
    {
        std::cerr<<"Error: Cannot set NULL CPDI driver to CPDIUpdateMethod, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    this->cpdi_driver_ = cpdi_driver;
}

//explicit instantiations
template class CPDIUpdateMethod<float,2>;
template class CPDIUpdateMethod<double,2>;
template class CPDIUpdateMethod<float,3>;
template class CPDIUpdateMethod<double,3>;

}  //end of namespace Physika
