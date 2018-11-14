/*
 * @file point_vector_attrib_config.h 
 * @Basic class PointVectorAttribConfig
 * @author Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#pragma once

#include <vector>
#include <glm/fwd.hpp>

#include "Physika_Core/Vectors/vector_3d.h"

namespace Physika{

class PointVectorAttribConfig
{
public:
    PointVectorAttribConfig();
    ~PointVectorAttribConfig();

    //disable copy
    PointVectorAttribConfig(const PointVectorAttribConfig &) = delete;
    PointVectorAttribConfig & operator = (const PointVectorAttribConfig &) = delete;

    //enable move
    PointVectorAttribConfig(PointVectorAttribConfig && rhs) noexcept;
    PointVectorAttribConfig & operator = (PointVectorAttribConfig && rhs) noexcept;

    template <typename Scalar>
    void setUniformPointVector(const Vector<Scalar, 3> & point_vector);
    
    template <typename Scalar>
    void setPointVectors(const std::vector<Vector<Scalar, 3>> & point_vectors);
    
    void setPointVectors(const std::vector<glm::vec3> & point_vectors);

    //Note: throw exception if point_vector size not match valid_num and not use uniform point_vector
    void config(unsigned int valid_num) const;
    unsigned int point_vectorNum() const;

private:
    void initPointVectorVBO();
    void destroyPointVectorVBO();

    void updatePointVectorVBOData(const std::vector<glm::vec3> & glm_col_vec);

private:
    unsigned int point_vector_num_ = 1;
    bool use_uniform_point_vector_ = true;
    Vector3f uniform_point_vector_ = {0, 1, 0};
    unsigned int point_vector_VBO_;
};
    
}//end of namespace Physika