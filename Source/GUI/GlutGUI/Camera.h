#pragma once
#include <vector>
#include "Core/Quaternion/quaternion.h"
#include "Core/Vector.h"

#define M_PI 3.14159265358979323846

namespace PhysIKA {
typedef Quaternion<float> Quat1f;

class Camera
{

public:
    Camera();
    ~Camera(){};

    void zoom(float amount);
    void setGL(float neardist, float fardist, float width, float height);

    Vector3f getViewDir() const;

    float getPixelArea() const;
    int   width() const;
    int   height() const;

    Vector3f getEye() const;
    void     setEyePostion(const Vector3f& pos);

    void  setFocus(float focus);
    float getFocus() const;

    void lookAt(const Vector3f& camPos, const Vector3f& center, const Vector3f& updir);

    void getCoordSystem(Vector3f& view, Vector3f& up, Vector3f& right) const;
    void rotate(Quat1f& rotquat);
    void localTranslate(const Vector3f translation);
    void translate(const Vector3f translation);

    /**
        * @brief Yaw around focus point. Rotation axis is m_worldUp.
        */
    void yawAroundFocus(float radian);

    /**
        * @brief Pitch around focus point. Rotation axis is m_right.
        */
    void pitchAroundFocus(float radian);

    /**
        * @brief Yaw around camera position. Rotation axis is m_worldUp.
        */
    void yaw(float radian);

    /**
        * @brief Pitch around camera position. Rotation axis is m_right.
        */
    void pitch(float radian);

    /**
        * @brief Update camera RIGHT, UP and VIEW direction.
        */
    void updateDir();

    void setWorldUp(const Vector3f& worldup)
    {
        m_worldUp = worldup;
    }
    void setLocalView(const Vector3f& localview)
    {
        m_localView = localview;
    }
    void setLocalRight(const Vector3f& localright)
    {
        m_localRight = localright;
    }
    void setLocalUp(const Vector3f& localup)
    {
        m_localUp = localup;
    }

private:
    void translateLight(const Vector3f translation);

private:
    float m_x;
    float m_y;

    float m_near;
    float m_far;
    float m_fov;

    int m_width;
    int m_height;

    float m_pixelarea;

    Vector3f m_eye;    // Camera position.
    Vector3f m_light;  // Light position.

    // Camera can rotate around focus point.
    // Focus point = m_eye + m_view * focus.
    float m_focus;

    Quaternionf m_cameraRot;  // Camera rotation quaternion.
    Vector3f    m_rotation_axis;
    float       m_rotation;

    Vector3f m_right;  // Camera right direction in world frame.
    Vector3f m_up;     // Camera up direction in world frame.
    Vector3f m_view;   // Camera view direction in world frame.

    Vector3f m_worldUp;     // World up direction.
    Vector3f m_localView;   // Camera view direction in camera frame.
    Vector3f m_localRight;  // Camera right direction in camera frame.
    Vector3f m_localUp;     // Camera up direction in camera frame.
};

}  // namespace PhysIKA
