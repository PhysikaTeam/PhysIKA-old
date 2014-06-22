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
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Render/Color/color.h"

namespace Physika{

/*
 * openGLVertex<Scalar,Dim>(const Vector<Scalar,Dim> &):
 * replacement for glVertex2f, glVertex2d, glVertex3f, glVertex3d
 */
template <typename Scalar,int Dim>
inline void openGLVertex(const Vector<Scalar,Dim> &vec)
{
}
template<> inline void openGLVertex(const Vector<float,2> &vec)
{
    glVertex2f(vec[0],vec[1]);
}
template<> inline void openGLVertex(const Vector<double,2> &vec)
{
    glVertex2d(vec[0],vec[1]);
}
template<> inline void openGLVertex(const Vector<float,3> &vec)
{
    glVertex3f(vec[0],vec[1],vec[2]);
}
template<> inline void openGLVertex(const Vector<double,3> &vec)
{
    glVertex3d(vec[0],vec[1],vec[2]);
}

/*
 * openGLColor<Scalar>{34}(const Color<Scalar> &):
 * replacement for glColor{34}b, glColor{34}s, glColor{34}i, glColor{34}f, glColor{34}d, 
 * glColor{34}ub, glColor{34}us, glColor{34}ui
 */
template <typename Scalar>
inline void openGLColor3(const Color<Scalar> &color)
{
}
template <typename Scalar>
inline void openGLColor4(const Color<Scalar> &color)
{
}
template<> inline void openGLColor3<char>(const Color<char> &color)
{
    glColor3b(color.redChannel(),color.greenChannel(),color.blueChannel());
}
template<> inline void openGLColor4<char>(const Color<char> &color)
{
    glColor4b(color.redChannel(),color.greenChannel(),color.blueChannel(),color.alphaChannel());
}
template<> inline void openGLColor3<short>(const Color<short> &color)
{
    glColor3s(color.redChannel(),color.greenChannel(),color.blueChannel());
}
template<> inline void openGLColor4<short>(const Color<short> &color)
{
    glColor4s(color.redChannel(),color.greenChannel(),color.blueChannel(),color.alphaChannel());
}
template<> inline void openGLColor3<int>(const Color<int> &color)
{
    glColor3i(color.redChannel(),color.greenChannel(),color.blueChannel());
}
template<> inline void openGLColor4<int>(const Color<int> &color)
{
    glColor4i(color.redChannel(),color.greenChannel(),color.blueChannel(),color.alphaChannel());
}
template<> inline void openGLColor3<float>(const Color<float> &color)
{
    glColor3f(color.redChannel(),color.greenChannel(),color.blueChannel());
}
template<> inline void openGLColor4<float>(const Color<float> &color)
{
    glColor4f(color.redChannel(),color.greenChannel(),color.blueChannel(),color.alphaChannel());
}
template<> inline void openGLColor3<double>(const Color<double> &color)
{
    glColor3d(color.redChannel(),color.greenChannel(),color.blueChannel());
}
template<> inline void openGLColor4<double>(const Color<double> &color)
{
    glColor4d(color.redChannel(),color.greenChannel(),color.blueChannel(),color.alphaChannel());
}
template<> inline void openGLColor3<unsigned char>(const Color<unsigned char> &color)
{
    glColor3ub(color.redChannel(),color.greenChannel(),color.blueChannel());
}
template<> inline void openGLColor4<unsigned char>(const Color<unsigned char> &color)
{
    glColor4ub(color.redChannel(),color.greenChannel(),color.blueChannel(),color.alphaChannel());
}
template<> inline void openGLColor3<unsigned short>(const Color<unsigned short> &color)
{
    glColor3us(color.redChannel(),color.greenChannel(),color.blueChannel());
}
template<> inline void openGLColor4<unsigned short>(const Color<unsigned short> &color)
{
    glColor4us(color.redChannel(),color.greenChannel(),color.blueChannel(),color.alphaChannel());
}
template<> inline void openGLColor3<unsigned int>(const Color<unsigned int> &color)
{
    glColor3ui(color.redChannel(),color.greenChannel(),color.blueChannel());
}
template<> inline void openGLColor4<unsigned int>(const Color<unsigned int> &color)
{
    glColor4ui(color.redChannel(),color.greenChannel(),color.blueChannel(),color.alphaChannel());
}

/*
 * openGLNormal<Scalar>(const Vector<Scalar,3> &):
 * replacement for glNormal3f, glNormal3d
 */
template <typename Scalar>
inline void openGLNormal(const Vector<Scalar,3> &normal)
{
}
template<> inline void openGLNormal(const Vector<float,3> &normal)
{
    glNormal3f(normal[0],normal[1],normal[2]);
}
template<> inline void openGLNormal(const Vector<double,3> &normal)
{
    glNormal3d(normal[0],normal[1],normal[2]);
}

/*
 * openGLMaterial<Scalar>(GLenum face, GLenum pname, Scalar param):
 * replacement for glMaterialf, glMateriali
 */
template <typename Scalar>
inline void openGLMaterial(GLenum face, GLenum pname, Scalar param)
{
}
template<> inline void openGLMaterial(GLenum face, GLenum pname, float param)
{
    glMaterialf(face,pname,param);
}
template<> inline void openGLMaterial(GLenum face, GLenum pname, int param)
{
    glMateriali(face,pname,param);
}

/*
 * openGLTranslate<Scalar>(const Vector<Scalar,3> &):
 * replacement for glTranslatef, glTranslated
 */
template <typename Scalar>
inline void openGLTranslate(const Vector<Scalar,3> &vec)
{
}
template<> inline void openGLTranslate(const Vector<float,3> &vec)
{
    glTranslatef(vec[0],vec[1],vec[2]);
}
template<> inline void openGLTranslate(const Vector<double,3> &vec)
{
    glTranslated(vec[0],vec[1],vec[2]);
}

/*
 * openGLRotate<Scalar>(Scalar angle, Scalar x, Scalar y, Scalar z):
 * replacement for glRotatef, glRotated
 */
template <typename Scalar>
inline void openGLRotate(Scalar angle, Scalar x, Scalar y, Scalar z)
{
}
template<> inline void openGLRotate(float angle, float x, float y, float z)
{
    glRotatef(angle,x,y,z);
}
template<> inline void openGLRotate(double angle, double x, double y, double z)
{
    glRotated(angle,x,y,z);
}

/*
 * openGLScale<Scalar>(const Vector<Scalar,3> &):
 * replacement for glScalef, glScaled
 */
template <typename Scalar>
inline void openGLScale(const Vector<Scalar,3> &vec)
{
}
template<> inline void openGLScale(const Vector<float,3> &vec)
{
    glScalef(vec[0],vec[1],vec[2]);
}
template<> inline void openGLScale(const Vector<double,3> &vec)
{
    glScaled(vec[0],vec[1],vec[2]);
}

/*
 * openGLLightModel<Scalar>(GLenum pname, Scalar param):
 * replacement for glLightModelf, glLightModeli
 */
template <typename Scalar>
inline void openGLLightModel(GLenum pname, Scalar param)
{
}
template<> inline void openGLLightModel(GLenum pname, float param)
{
    glLightModelf(pname,param);
}
template<> inline void openGLLightModel(GLenum pname, int param)
{
    glLightModeli(pname,param);
}

/*
 * openGLTexParameter<Scalar>(GLenum target, GLenum pname, Scalar param):
 * replacement for glTexParameteri, glTexParameterf
 */
template <typename Scalar>
inline void openGLTexParameter(GLenum target, GLenum pname, Scalar param)
{
}
template<> inline void openGLTexParameter(GLenum target, GLenum pname, float param)
{
    glTexParameterf(target,pname,param);
}
template<> inline void openGLTexParameter(GLenum target, GLenum pname, int param)
{
    glTexParameteri(target,pname,param);
}

/*
 * openGLTexCoord<Scalar,Dim>(const Vector<Scalar,Dim> &):
 * replacement for glTexCoord2f, glTexCoord2d, glTexCoord3f, glTexCoord3d
 */
template <typename Scalar, int Dim>
inline void openGLTexCoord(const Vector<Scalar,Dim> &vec)
{
}
template<> inline void openGLTexCoord(const Vector<float,2> &vec)
{
    glTexCoord2f(vec[0],vec[1]);
}
template<> inline void openGLTexCoord(const Vector<double,2> &vec)
{
    glTexCoord2d(vec[0],vec[1]);
}
template<> inline void openGLTexCoord(const Vector<float,3> &vec)
{
    glTexCoord3f(vec[0],vec[1],vec[2]);
}
template<> inline void openGLTexCoord(const Vector<double,3> &vec)
{
    glTexCoord3d(vec[0],vec[1],vec[2]);
}

} //end of namespace Physika

#endif //PHYSIKA_RENDER_OPENGL_PRIMITIVES_OPENGL_PRIMITIVES_H_
