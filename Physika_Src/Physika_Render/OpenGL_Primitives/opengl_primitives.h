/*
 * @file opengl_primitives.h 
 * @Brief Template wrappers for type-dependent OpenGL primitives
 *        If a render is type-dependent, these primitives should be used instead of
 *        directly casting to one type
 * @Note: If the openGL function you need to use is not here, you can add it here
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

#ifndef PHYSIKA_RENDER_OPENGL_PRIMITIVES_OPENGL_PRIMITIVES_H_
#define PHYSIKA_RENDER_OPENGL_PRIMITIVES_OPENGL_PRIMITIVES_H_

#include <GL/gl.h>
#include <iostream>
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Render/Color/color.h"
#include "Physika_Core/Matrices/matrix_4x4.h"
#include "Physika_Core/Matrices/matrix_3x3.h"

namespace Physika{

/*
 * openGLVertex(const Vector<Scalar,Dim> &):
 * replacement for glVertex2f, glVertex2d, glVertex3f, glVertex3d
 */
inline void openGLVertex(const Vector<float,2> &vec)
{
    glVertex2f(vec[0],vec[1]);
}
inline void openGLVertex(const Vector<double,2> &vec)
{
    glVertex2d(vec[0],vec[1]);
}
inline void openGLVertex(const Vector<float,3> &vec)
{
    glVertex3f(vec[0],vec[1],vec[2]);
}
inline void openGLVertex(const Vector<double,3> &vec)
{
    glVertex3d(vec[0],vec[1],vec[2]);
}

/*
 * openGLColor{34}(const Color<Scalar> &):
 * replacement for glColor{34}b, glColor{34}s, glColor{34}i, glColor{34}f, glColor{34}d, 
 * glColor{34}ub, glColor{34}us, glColor{34}ui
 */
inline void openGLColor3(const Color<signed char> &color)
{
    glColor3b(color.redChannel(),color.greenChannel(),color.blueChannel());
}
inline void openGLColor4(const Color<signed char> &color)
{
    glColor4b(color.redChannel(),color.greenChannel(),color.blueChannel(),color.alphaChannel());
}
inline void openGLColor3(const Color<short> &color)
{
    glColor3s(color.redChannel(),color.greenChannel(),color.blueChannel());
}
inline void openGLColor4(const Color<short> &color)
{
    glColor4s(color.redChannel(),color.greenChannel(),color.blueChannel(),color.alphaChannel());
}
inline void openGLColor3(const Color<int> &color)
{
    glColor3i(color.redChannel(),color.greenChannel(),color.blueChannel());
}
inline void openGLColor4(const Color<int> &color)
{
    glColor4i(color.redChannel(),color.greenChannel(),color.blueChannel(),color.alphaChannel());
}
inline void openGLColor3(const Color<float> &color)
{
    glColor3f(color.redChannel(),color.greenChannel(),color.blueChannel());
}
inline void openGLColor4(const Color<float> &color)
{
    glColor4f(color.redChannel(),color.greenChannel(),color.blueChannel(),color.alphaChannel());
}
inline void openGLColor3(const Color<double> &color)
{
    glColor3d(color.redChannel(),color.greenChannel(),color.blueChannel());
}
inline void openGLColor4(const Color<double> &color)
{
    glColor4d(color.redChannel(),color.greenChannel(),color.blueChannel(),color.alphaChannel());
}
inline void openGLColor3(const Color<unsigned char> &color)
{
    glColor3ub(color.redChannel(),color.greenChannel(),color.blueChannel());
}
inline void openGLColor4(const Color<unsigned char> &color)
{
    glColor4ub(color.redChannel(),color.greenChannel(),color.blueChannel(),color.alphaChannel());
}
inline void openGLColor3(const Color<unsigned short> &color)
{
    glColor3us(color.redChannel(),color.greenChannel(),color.blueChannel());
}
inline void openGLColor4(const Color<unsigned short> &color)
{
    glColor4us(color.redChannel(),color.greenChannel(),color.blueChannel(),color.alphaChannel());
}
inline void openGLColor3(const Color<unsigned int> &color)
{
    glColor3ui(color.redChannel(),color.greenChannel(),color.blueChannel());
}
inline void openGLColor4(const Color<unsigned int> &color)
{
    glColor4ui(color.redChannel(),color.greenChannel(),color.blueChannel(),color.alphaChannel());
}

/*
 * openGLNormal(const Vector<Scalar,3> &):
 * replacement for glNormal3f, glNormal3d
 */
inline void openGLNormal(const Vector<float,3> &normal)
{
    glNormal3f(normal[0],normal[1],normal[2]);
}
inline void openGLNormal(const Vector<double,3> &normal)
{
    glNormal3d(normal[0],normal[1],normal[2]);
}

/*
 * openGLMaterial(GLenum face, GLenum pname, Scalar param):
 * replacement for glMaterialf, glMateriali
 */
inline void openGLMaterial(GLenum face, GLenum pname, float param)
{
    glMaterialf(face,pname,param);
}
inline void openGLMaterial(GLenum face, GLenum pname, int param)
{
    glMateriali(face,pname,param);
}

/*
 * openGLMaterialv(GLenum face, GLenum pname, const Scalar *param):
 * replacement for glMaterialfv, glMaterialiv
 */
inline void openGLMaterialv(GLenum face, GLenum pname, const float *param)
{
    glMaterialfv(face,pname,param);
}
inline void openGLMaterialv(GLenum face, GLenum pname, const int *param)
{
    glMaterialiv(face,pname,param);
}

/*
 * openGLTranslate(const Vector<Scalar,2> &):
 * openGLTranslate(const Vector<Scalar,3> &):
 * replacement for glTranslatef, glTranslated
 */
inline void openGLTranslate(const Vector<float,2> &vec)
{
    glTranslatef(vec[0],vec[1],0);
}
inline void openGLTranslate(const Vector<double,2> &vec)
{
    glTranslated(vec[0],vec[1],0);
}
inline void openGLTranslate(const Vector<float,3> &vec)
{
    glTranslatef(vec[0],vec[1],vec[2]);
}
inline void openGLTranslate(const Vector<double,3> &vec)
{
    glTranslated(vec[0],vec[1],vec[2]);
}

/*
 * openGLRotate(Scalar angle, Scalar x, Scalar y, Scalar z):
 * replacement for glRotatef, glRotated
 */
inline void openGLRotate(float angle, float x, float y, float z)
{
    glRotatef(angle,x,y,z);
}
inline void openGLRotate(double angle, double x, double y, double z)
{
    glRotated(angle,x,y,z);
}

/*
 * openGLScale(const Vector<Scalar,3> &):
 * replacement for glScalef, glScaled
 */
inline void openGLScale(const Vector<float,3> &vec)
{
    glScalef(vec[0],vec[1],vec[2]);
}
inline void openGLScale(const Vector<double,3> &vec)
{
    glScaled(vec[0],vec[1],vec[2]);
}

/*
 * openGLLight(GLenum light, GLenum pname, Scalar param):
 * replacement for glLighti, glLightf
 */
inline void openGLLight(GLenum light, GLenum pname,int param)
{
    glLighti(light, pname, param);
}
inline void openGLLight(GLenum light, GLenum pname,float param)
{
    glLightf(light, pname, param);
}
inline void openGLLight(GLenum light, GLenum pname,double param)
{
    glLightf(light, pname, static_cast<float>(param));
}
/*
 * openGLLightv(GLenum light, GLenum pname, Color/Vector<Scalar> param):
 * replacement for glLightiv, glLightfv
 */
template <typename ColorType>
inline void openGLLightv(GLenum light, GLenum pname, const Color<ColorType>& color)
{
    Color<float> temp_color = color.template convertColor<float>();
    float param[4];
    param[0] = temp_color.redChannel();
    param[1] = temp_color.greenChannel();
    param[2] = temp_color.blueChannel();
    param[3] = temp_color.alphaChannel();
    glLightfv(light, pname, param);
}

/// warning: this function is defined particularly to specify GL_SPOT_DIRECTION and GL_POSITION
template <int Dim>
inline void openGLLightv(GLenum light, GLenum pname,const Vector<float, Dim>& pos_dir)
{
	float param[Dim];
	for(unsigned int i=0; i<Dim; i++)
	{
		param[i] = pos_dir[i];
	}
	glLightfv(light, pname, param);
}
template <int Dim>
inline void openGLLightv(GLenum light, GLenum pname,const Vector<double, Dim>& pos_dir)
{
	float param[Dim];
	for(unsigned int i=0; i<Dim; i++)
	{
		param[i] = static_cast<float>(pos_dir[i]);
	}
	glLightfv(light, pname, param);
}
template <int Dim>
inline void openGLLightv(GLenum light, GLenum pname,const Vector<int, Dim>& pos_dir)
{
	int param[Dim];
	for(unsigned int i=0; i<Dim; i++)
	{
		param[i] = pos_dir[i];
	}
	glLightiv(light, pname, param);
}

/*
 * openGLLightModel(GLenum pname, Scalar param):
 * replacement for glLightModelf, glLightModeli
 */

inline void openGLLightModel(GLenum pname, float param)
{
    glLightModelf(pname,param);
}
inline void openGLLightModel(GLenum pname, double param)
{
    glLightModelf(pname, static_cast<float>(param));
}
inline void openGLLightModel(GLenum pname, int param)
{
    glLightModeli(pname,param);
}
inline void openGLLightModel(GLenum pname, unsigned int param)
{
    glLightModeli(pname,param);
}

/*
 * openGLLightModelv(GLenum pname, const Scalar *param):
 * replacement for glLightModelfv, glLightModeliv
 */
/// warning: this function is designed particularly to specify GL_LIGHT_MODEL_AMBIENT
template <typename ColorType>
inline void openGLLightModelAMBient(const Color<ColorType>& color)
{
    Color<float> temp_color = color.template convertColor<float>();
    float param[4];
    param[0] = temp_color.redChannel();
    param[1] = temp_color.greenChannel();
    param[2] = temp_color.blueChannel();
    param[3] = temp_color.alphaChannel();
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, param);
}

/*
 * openGLTexParameter(GLenum target, GLenum pname, Scalar param):
 * replacement for glTexParameteri, glTexParameterf
 */
inline void openGLTexParameter(GLenum target, GLenum pname, float param)
{
    glTexParameterf(target,pname,param);
}
inline void openGLTexParameter(GLenum target, GLenum pname, double param)
{
    glTexParameterf(target,pname,static_cast<float>(param));
}
inline void openGLTexParameter(GLenum target, GLenum pname, int param)
{
    glTexParameteri(target,pname,param);
}

/*
 * openGLTexCoord(const Vector<Scalar,Dim> &):
 * replacement for glTexCoord2f, glTexCoord2d, glTexCoord3f, glTexCoord3d
 */
inline void openGLTexCoord(const Vector<float,2> &vec)
{
    glTexCoord2f(vec[0],vec[1]);
}
inline void openGLTexCoord(const Vector<double,2> &vec)
{
    glTexCoord2d(vec[0],vec[1]);
}
inline void openGLTexCoord(const Vector<float,3> &vec)
{
    glTexCoord3f(vec[0],vec[1],vec[2]);
}
inline void openGLTexCoord(const Vector<double,3> &vec)
{
    glTexCoord3d(vec[0],vec[1],vec[2]);
}

/*
 * openGLMultMatrix(const SquareMatrix<Scalar,3> &)
 * openGLMultMatrix(const SquareMatrix<Scalar,4> &):
 * replacement for glMultiMatrixf, glMultiMatrixd
 */

inline void openGLMultMatrix(const SquareMatrix<float,3> & matrix)
{
    float matrix_[16];

    for(unsigned int i=0; i<2; i++)
    {
        for(unsigned int j=0; j<3; j++)
        {
            matrix_[i*4+j] = matrix(j,i);
        }
        matrix_[i*4+3] = 0;
    }
    matrix_[8] = matrix_[9] = matrix_[10] = matrix_[11] = matrix_[14] = 0;
    matrix_[12] = matrix(0,2);
    matrix_[13] = matrix(1,2);
    matrix_[15] = matrix(2,2);

    glMultMatrixf(matrix_);

}
inline void openGLMultMatrix(const SquareMatrix<double,3> & matrix)
{
    double matrix_[16];

    for(unsigned int i=0; i<2; i++)
    {
        for(unsigned int j=0; j<3; j++)
        {
            matrix_[i*4+j] = matrix(j,i);
        }
        matrix_[i*4+3] = 0;
    }
    matrix_[8] = matrix_[9] = matrix_[10] = matrix_[11] = matrix_[14] = 0;
    matrix_[12] = matrix(0,2);
    matrix_[13] = matrix(1,2);
    matrix_[15] = matrix(2,2);

    glMultMatrixd(matrix_);


}
inline void openGLMultMatrix(const SquareMatrix<float,4> & matrix)
{
    float matrix_[16];
    for(unsigned int i=0; i<4; i++)
        for(unsigned int j=0; j<4; j++)
            matrix_[i*4+j] = matrix(j,i);
    glMultMatrixf(matrix_);
}
inline void openGLMultMatrix(const SquareMatrix<double,4> & matrix)
{
    double matrix_[16];
    for(unsigned int i=0; i<4; i++)
        for(unsigned int j=0; j<4; j++)
            matrix_[i*4+j] = matrix(j,i);
    glMultMatrixd(matrix_);
}

} //end of namespace Physika

#endif //PHYSIKA_RENDER_OPENGL_PRIMITIVES_OPENGL_PRIMITIVES_H_
