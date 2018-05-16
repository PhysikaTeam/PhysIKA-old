/*
 * @file tetrahedron_render_util.h 
 * @Basic class TetrahedronRenderUtil
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
#include <memory>
#include <glm/glm.hpp>

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Utilities/physika_exception.h"

#include "Physika_Render/Triangle_Render/triangle_render_util.h"


namespace Physika {

class TetrahedronGLCudaBuffer;

class TetrahedronRenderUtil
{
public:
    TetrahedronRenderUtil();
    ~TetrahedronRenderUtil();

    //disable copy
    TetrahedronRenderUtil(const TetrahedronRenderUtil &) = delete;
    TetrahedronRenderUtil & operator = (const TetrahedronRenderUtil &) = delete;

    template <typename Scalar>
    void setTetrahedrons(const std::vector<Vector<Scalar, 3>> & pos_vec, bool auto_compute_normal = true);

    template <typename Scalar>
    void setTetrahedrons(const std::vector<Vector<Scalar, 3>> & pos_vec, std::vector<unsigned int> & indices, bool auto_compute_normal = true);

    //Note: normals.size() = 4 * tet_num, per normal for one face
    template <typename Scalar>
    void setNormals(const std::vector<Vector<Scalar, 3>> & normals);

    TetrahedronGLCudaBuffer mapTetrahedronGLCudaBuffer(unsigned int tet_num);
    void unmapTetrahedronGLCudaBuffer();

    unsigned int tetrahedronNum() const;
    std::shared_ptr<TriangleRenderUtil> getInnerTriangleRenderUtil();

    void draw();

    void bindTetrahedronVAO();
    void unbindTetrahedronVAO();

private:
    std::shared_ptr<TriangleRenderUtil>  triangle_render_util_;
};

}//end of namespace Physika