/*
 * @file color_bar_render.cpp 
 * @Brief render of color bar.
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

#include <GL/freeglut.h>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"
#include "Physika_Render/ColorBar_Render/color_bar_render.h"

namespace Physika{

template <typename Scalar>
ColorBarRender<Scalar>::ColorBarRender()
    :color_bar_(NULL)
{

}

template <typename Scalar>
ColorBarRender<Scalar>::ColorBarRender(ColorBar<Scalar> * color_bar)
    :color_bar_(color_bar)
{

}

template <typename Scalar>
ColorBarRender<Scalar>::~ColorBarRender()
{

}

template <typename Scalar>
const ColorBar<Scalar> * ColorBarRender<Scalar>::colorBar() const
{
    return this->color_bar_;
}

template <typename Scalar>
void ColorBarRender<Scalar>::setColorBar(ColorBar<Scalar> * color_bar)
{
    if (color_bar == NULL)
    {
        std::cerr<<"can't set NULL color bar!\n";
        std::exit(EXIT_FAILURE);
    }
    this->color_bar_ = color_bar;
}

template <typename Scalar>
void ColorBarRender<Scalar>::render()
{
    const Vector<Scalar, 2> & pos = color_bar_->startPoint();
    
    // undefined in lib or dll?
    //glWindowPos2f(pos[0],pos[1]); 

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    unsigned int win_width = glutGet(GLUT_WINDOW_WIDTH);
    unsigned int win_height = glutGet(GLUT_WINDOW_HEIGHT);
    gluOrtho2D(0.0, (GLfloat)win_width, 0.0, (GLfloat)win_height);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glRasterPos2f(pos[0], pos[1]);

    unsigned int width = color_bar_->width();
    unsigned int height = color_bar_->height();
    float * pixels = new float[width*height*3];
    PHYSIKA_ASSERT(pixels);
    const ColorMap<Scalar> & color_map = color_bar_->colorMap();
    if (color_bar_->isHorizon() == false)
    {
        for (unsigned int i=0; i<height; i++)
        {
            const Color<Scalar> & color = color_map.colorViaRatio(static_cast<Scalar>(i)/height);
            Scalar r = color.redChannel();
            Scalar g = color.greenChannel();
            Scalar b = color.blueChannel();
            for (unsigned int j=0; j<width; j++)
            {
                pixels[(i*width+j)*3] = r;
                pixels[(i*width+j)*3+1] = g;
                pixels[(i*width+j)*3+2] = b;
            }
        }
        glDrawPixels(width, height, GL_RGB, GL_FLOAT, pixels);
    }
    else
    {
        for (unsigned int i=0; i<height; i++)
        {
            const Color<Scalar> & color = color_map.colorViaRatio(static_cast<Scalar>(i)/height);
            Scalar r = color.redChannel();
            Scalar g = color.greenChannel();
            Scalar b = color.blueChannel();
            for (unsigned int j=0; j<width; j++)
            {
                pixels[(j*height+i)*3] = r;
                pixels[(j*height+i)*3+1] = g;
                pixels[(j*height+i)*3+2] = b;
            }
        }
        glDrawPixels(height, width, GL_RGB, GL_FLOAT, pixels);
    }
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    delete [] pixels;
}

template <typename Scalar>
void ColorBarRender<Scalar>::drawBox(const Vector<Scalar, 3> & start_pos, Scalar dx, Scalar dy, Scalar dz, const Color<Scalar> & color)
{
    Vector<Scalar,3> position_0 = start_pos;
    Vector<Scalar,3> position_1 = start_pos+Vector<Scalar,3>(dx,0, 0);
    Vector<Scalar,3> position_2 = start_pos+Vector<Scalar,3>(dx,0, dz);
    Vector<Scalar,3> position_3 = start_pos+Vector<Scalar,3>(0, 0, dz);
    Vector<Scalar,3> position_4 = start_pos+Vector<Scalar,3>(0, dy, 0);
    Vector<Scalar,3> position_5 = start_pos+Vector<Scalar,3>(dx,dy,0);
    Vector<Scalar,3> position_6 = start_pos+Vector<Scalar,3>(dx,dy,dz);
    Vector<Scalar,3> position_7 = start_pos+Vector<Scalar,3>(0 ,dy,dz);

    openGLColor3(color);

    glBegin(GL_POLYGON);
    openGLVertex(position_0);
    openGLVertex(position_3);
    openGLVertex(position_2);
    openGLVertex(position_1);
    glEnd();

    glBegin(GL_POLYGON);
    openGLVertex(position_4);
    openGLVertex(position_5);
    openGLVertex(position_6);
    openGLVertex(position_7);
    glEnd();

    glBegin(GL_POLYGON);
    openGLVertex(position_0);
    openGLVertex(position_1);
    openGLVertex(position_5);
    openGLVertex(position_4);
    glEnd();

    glBegin(GL_POLYGON);
    openGLVertex(position_3);
    openGLVertex(position_7);
    openGLVertex(position_6);
    openGLVertex(position_2);
    glEnd();

    glBegin(GL_POLYGON);
    openGLVertex(position_4);
    openGLVertex(position_7);
    openGLVertex(position_3);
    openGLVertex(position_0);
    glEnd();

    glBegin(GL_POLYGON);
    openGLVertex(position_1);
    openGLVertex(position_2);
    openGLVertex(position_6);
    openGLVertex(position_5);
    glEnd();
}

//explicit instantiation
template class ColorBarRender<float>;
template class ColorBarRender<double>;

}// end of namespace Physika