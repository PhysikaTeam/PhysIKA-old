/*
 * @file array.h 
 * @brief array class. Design for general using.
 * @author Sheng Yang
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHSYIKA_CORE_ARRAY_ARRAY_H_
#define PHSYIKA_CORE_ARRAY_ARRAY_H_

#include "Physika_Core/Utilities/global_config.h"
#include "Physika_Core/Vectors/vector_3d.h"

namespace Physika{

template <typename Scalar>
class Array
{
public:    
    /* Constructors */
    Array();
    Array(Scalar* data, unsigned int element_cout);
    Array(const Array<Scalar>& );
    ~Array();
    
    /* Assignment operators */
    Array<Scalar>& operator = (const Array<Scalar>& arr);

    /* Get and Set functions */
    inline unsigned int element_cout() const { return element_cout_; }
    inline Scalar* data() const { return data_; }

    /* Special functions */
    void reset(unsigned int count);
    void setSpace(unsigned int count);
    void zero();

    /* Operator overloading */
    inline Scalar & operator[] (unsigned int id){ assert(id >= 0 && id <= element_cout_); return data_[id]; }


protected:
    void allocate();
    void release();

    unsigned int element_cout_;
    Scalar * data_;

};


template <typename Scalar>
std::ostream& operator<< (std::ostream &s, const Array<Scalar> &arr)
{
    for(size_t i = 0; i < arr.element_cout(); i++)
    {
        if(i == 0)
            s<<arr[i];
        s<<", "<<arr[i];
    }
    s<<std::endl;
    return s; 
}




//convenient typedefs
typedef Array<int> Arrayi;
typedef Array<float> Arrayf;
typedef Array<double> Arrayd;
typedef Array<Vector3f> Arrayv3f;
typedef Array<Vector3d> Arrayv3d;


}//end of namespace Physika

#endif //PHSYIKA_CORE_ARRAY_ARRAY_H_
