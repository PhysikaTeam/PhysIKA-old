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
#include <limits>
#include <iostream>
#include <vector>
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Utilities/physika_assert.h"
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
    PHYSIKA_ASSERT(this->cpdi_driver_);
    std::vector<Vector<Scalar,Dim> > domain_corner;
    unsigned int corner_num = Dim==2 ? 4 : 8;
    for(unsigned int i = 0; i < cpdi_driver_->particleNum(); ++i)
    {
        Vector<Scalar,Dim> min_corner((std::numeric_limits<Scalar>::max)());
        Vector<Scalar,Dim> max_corner((std::numeric_limits<Scalar>::min)());
        cpdi_driver_->initialParticleDomain(i,domain_corner);
        PHYSIKA_ASSERT(domain_corner.size()==corner_num);
        for(unsigned int j = 0; j < domain_corner.size(); ++j)
            for(unsigned int dim = 0; dim < Dim; ++dim)
            {
                if(domain_corner[j][dim]<min_corner[dim])
                    min_corner[dim] = domain_corner[j][dim];
                if(domain_corner[j][dim]>max_corner[dim])
                    max_corner[dim] = domain_corner[j][dim];
            }
//TO DO

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
