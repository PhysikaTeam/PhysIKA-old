#pragma once

#include "../math/math.h"

class ParticleSolver
{
public:
    vecf3   _pos;
    vecf4   _color;
    cfloat3 _xMin;
    cfloat3 _xMax;
    float   _dt;

public:
    virtual vecf3& getPos()
    {
        return _pos;
    };
    virtual vecf4& getColor()
    {
        return _color;
    }
    virtual cfloat3 getXmin()
    {
        return _xMin;
    }
    virtual cfloat3 getXmax()
    {
        return _xMax;
    }
    virtual float getDt()
    {
        return _dt;
    }
    virtual void prepareRenderData()
    {
    }
    virtual void dumpRenderData()
    {
    }

    virtual void keyDown(int key) {}
    virtual void keyUp(int key) {}

    virtual void step()
    {
    }

    virtual void handleKeyEvent(char key)
    {
    }
    virtual void eval(const char* expression)
    {
    }
};