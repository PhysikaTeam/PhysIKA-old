/*
 * @file point_render_util.cpp
 * @Basic class PointRenderUtil
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

#include "Physika_Render/OpenGL_Primitives/glew_utilities.h"
#include "point_render_util.h"

namespace Physika{

PointRenderUtil::PointRenderUtil()
{
    this->initVAOAndVBO();
}

PointRenderUtil::~PointRenderUtil()
{
    this->destoryVAOAndVBO();
}

template <typename Scalar, int Dim>
void PointRenderUtil::setPoints(const std::vector<Vector<Scalar, Dim>> & pos_vec)
{
    if (Dim != 2 && Dim != 3)
        throw PhysikaException("error: require Dim == 2 || 3!");

    this->point_num_ = pos_vec.size();

    std::vector<glm::vec3> glm_pos_vec(pos_vec.size());
    for(unsigned int i = 0; i < pos_vec.size(); ++i)
    {
        if (Dim == 2)
            glm_pos_vec[i] = { pos_vec[i][0], pos_vec[i][1], 0 };
        else
            glm_pos_vec[i] = { pos_vec[i][0], pos_vec[i][1], pos_vec[i][2] };
    }

    this->updatePositionVBOData(glm_pos_vec);

}

unsigned int PointRenderUtil::pointNum() const
{
    return point_num_;
}

void PointRenderUtil::draw()
{
    glBindVertexArray(point_VAO_);
    glDrawArrays(GL_POINTS, 0, point_num_);
    glBindVertexArray(0);
}

void PointRenderUtil::bindPointVAO()
{
    glVerify(glBindVertexArray(point_VAO_));
}

void PointRenderUtil::unbindPointVAO()
{
    glVerify(glBindVertexArray(0));
}


void PointRenderUtil::initVAOAndVBO()
{
    glVerify(glGenVertexArrays(1, &point_VAO_));
    glVerify(glBindVertexArray(point_VAO_));

    glGenBuffers(1, &pos_VBO_);
    glBindBuffer(GL_ARRAY_BUFFER, pos_VBO_);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glVerify(glBindVertexArray(0));
}

void PointRenderUtil::destoryVAOAndVBO()
{
    glDeleteVertexArrays(1, &point_VAO_);
    glDeleteBuffers(1, &pos_VBO_);
}

void PointRenderUtil::updatePositionVBOData(const std::vector<glm::vec3> & glm_pos_vec)
{
    glBindBuffer(GL_ARRAY_BUFFER, pos_VBO_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * glm_pos_vec.size(), glm_pos_vec.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

template void PointRenderUtil::setPoints<float, 2>(const std::vector<Vector<float, 2>> & pos_vec);
template void PointRenderUtil::setPoints<float, 3>(const std::vector<Vector<float, 3>> & pos_vec);
template void PointRenderUtil::setPoints<double, 2>(const std::vector<Vector<double, 2>> & pos_vec);
template void PointRenderUtil::setPoints<double, 3>(const std::vector<Vector<double, 3>> & pos_vec);

}
