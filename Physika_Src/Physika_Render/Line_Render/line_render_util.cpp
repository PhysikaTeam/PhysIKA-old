/*
 * @file line_render_util.cpp 
 * @Basic class LineRenderUtil
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
#include "line_render_util.h"

namespace Physika{

LineRenderUtil::LineRenderUtil()
{
    this->initVAOAndVBO();
}

LineRenderUtil::~LineRenderUtil()
{
    this->destoryVAOAndVBO();
}

unsigned int LineRenderUtil::lineNum() const
{
    return line_num_;
}

template <typename Scalar, int Dim>
void LineRenderUtil::setLinePairs(const std::vector<Vector<Scalar, Dim>> & line_pairs)
{
    if (line_pairs.size() % 2 != 0)
        throw PhysikaException("error: line_pairs.size() % 2 != 0!");

    std::vector<glm::vec3> glm_pos_vec(line_pairs.size());
    for (unsigned int i = 0; i < line_pairs.size(); ++i)
    {
        if (Dim == 2)
            glm_pos_vec[i] = { line_pairs[i][0], line_pairs[i][1], 0.0 };
        else
            glm_pos_vec[i] = { line_pairs[i][0], line_pairs[i][1], line_pairs[i][2] };
    }

    line_num_ = glm_pos_vec.size() / 2;
    this->updateLineVBOData(glm_pos_vec);
}

template <typename Scalar, int Dim>
void LineRenderUtil::setLineStrips(const std::vector<Vector<Scalar, Dim>> & line_strips)
{
    if (line_strips.size() < 2)
        throw PhysikaException("error: line_strips.size() < 2!");

    std::vector<glm::vec3> glm_pos_vec;
    for (unsigned int i = 0; i < line_strips.size() - 1; ++i)
    {
        glm::vec3 fir_pos, sec_pos;
        if (Dim == 2)
        {
            fir_pos = { line_strips[i][0], line_strips[i][1], 0.0 };
            sec_pos = { line_strips[i + 1][0], line_strips[i + 1][1], 0.0 };
        }
        else
        {
            fir_pos = { line_strips[i][0], line_strips[i][1], line_strips[i][2] };
            sec_pos = { line_strips[i + 1][0], line_strips[i + 1][1], line_strips[i + 1][2] };
        }

        glm_pos_vec.push_back(fir_pos);
        glm_pos_vec.push_back(sec_pos);
    }

    line_num_ = glm_pos_vec.size() / 2;
    this->updateLineVBOData(glm_pos_vec);
}

template <typename Scalar, int Dim>
void LineRenderUtil::setLines(const std::vector<Vector<Scalar, Dim>> & pos_vec, std::vector<unsigned int> & indices)
{
    if (indices.size() < 2 || indices.size() % 2 != 0)
        throw PhysikaException("error: indices.size() < 2 || indices.size() % 2 != 0!");

    std::vector<glm::vec3> glm_pos_vec(indices.size());
    for (unsigned int i = 0; i < indices.size(); ++i)
    {
        unsigned int index = indices[i];

        if (Dim == 2)
            glm_pos_vec[i] = { pos_vec[index][0], pos_vec[index][1], 0.0 };
        else
            glm_pos_vec[i] = { pos_vec[index][0], pos_vec[index][1], pos_vec[index][2] };
    }

    line_num_ = glm_pos_vec.size() / 2;
    this->updateLineVBOData(glm_pos_vec);
}

void LineRenderUtil::draw()
{
    glBindVertexArray(line_VAO_);
    glDrawArrays(GL_LINES, 0, 2 * line_num_);
    glBindVertexArray(0);
}

void LineRenderUtil::bindLineVAO()
{
    glVerify(glBindVertexArray(line_VAO_));
}

void LineRenderUtil::unbindLineVAO()
{
    glVerify(glBindVertexArray(0));
}
 
void LineRenderUtil::initVAOAndVBO()
{
    glVerify(glGenVertexArrays(1, &line_VAO_));
    glVerify(glBindVertexArray(line_VAO_));

    glGenBuffers(1, &line_VBO_);
    glBindBuffer(GL_ARRAY_BUFFER, line_VBO_);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glVerify(glBindVertexArray(0));
}

void LineRenderUtil::destoryVAOAndVBO()
{
    glDeleteVertexArrays(1, &line_VAO_);
    glDeleteBuffers(1, &line_VBO_);
}

void LineRenderUtil::updateLineVBOData(const std::vector<glm::vec3> & glm_pos_vec)
{
    glBindBuffer(GL_ARRAY_BUFFER, line_VBO_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * glm_pos_vec.size(), glm_pos_vec.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

//explicit instantiation
template void LineRenderUtil::setLinePairs<float, 2>(const std::vector<Vector<float, 2>> &);
template void LineRenderUtil::setLinePairs<float, 3>(const std::vector<Vector<float, 3>> &);
template void LineRenderUtil::setLinePairs<double, 2>(const std::vector<Vector<double, 2>> &);
template void LineRenderUtil::setLinePairs<double, 3>(const std::vector<Vector<double, 3>> &);

template void LineRenderUtil::setLineStrips<float, 2>(const std::vector<Vector<float, 2>> &);
template void LineRenderUtil::setLineStrips<float, 3>(const std::vector<Vector<float, 3>> &);
template void LineRenderUtil::setLineStrips<double, 2>(const std::vector<Vector<double, 2>> &);
template void LineRenderUtil::setLineStrips<double, 3>(const std::vector<Vector<double, 3>> &);

template void LineRenderUtil::setLines<float, 2>(const std::vector<Vector<float, 2>> &, std::vector<unsigned int> &);
template void LineRenderUtil::setLines<float, 3>(const std::vector<Vector<float, 3>> &, std::vector<unsigned int> &);
template void LineRenderUtil::setLines<double, 2>(const std::vector<Vector<double, 2>> &, std::vector<unsigned int> &);
template void LineRenderUtil::setLines<double, 3>(const std::vector<Vector<double, 3>> &, std::vector<unsigned int> &);
    
}//end of namespace Physika