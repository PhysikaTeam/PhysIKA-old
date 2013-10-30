/*
 * @file point_render.h 
 * @Basic render of point, it is used to draw the simulate result.
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

#ifndef PHYSIKA_RENDER_POINT_RENDER_POINT_RENDER_H_
#define PHYSIKA_RENDER_POINT_RENDER_POINT_RENDER_H_

#include "Physika_Render/Render_Base/render_base.h"
#include "Physika_Core/Vectors/vector_3d.h"

namespace Physika{

template <typename Scalar>
class PointRender: public RenderBase
{
public:
    /* Constructions */
    PointRender();
    PointRender(Vector<Scalar,3> * points, int num_of_point);
    ~PointRender(void);

    /* Get and Set */
    inline Vector<Scalar,3>* points() { return points_; }
    inline int num_of_point() { return num_of_point_; }
    inline void setNbPoint(int num_of_point) { num_of_point_ = num_of_point; }

    /* Render */
    virtual void render();


protected:
    /* Render data */
    int num_of_point_;
    Vector<Scalar,3> *points_;

};

} //end of namespace Physika

#endif //PHYSIKA_RENDER_POINT_RENDER_POINT_RENDER_H_
