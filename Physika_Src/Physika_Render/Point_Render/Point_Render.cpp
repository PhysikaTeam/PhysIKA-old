/*
 * @file point_render.cpp 
 * @Basic render of point, it is used to draw the simulate result of points.
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

#include "Physika_Render/Point_Render/Point_Render.h"

namespace Physika{
	
PointRender::PointRender(): num_of_point_(0)
{
	points_ = NULL;
}

PointRender::PointRender(Vector3f * points, int num_of_point):num_of_point_(num_of_point)
{
   points_ = points;
}

PointRender::~PointRender(void)
{

}

void PointRender::render()
{
    //To render these points in a GLUI panel based screen.

}



} //end of namespace Physika
