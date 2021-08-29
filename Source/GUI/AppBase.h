#pragma once
#include <vector>
#include <algorithm>
#include <iostream>
#include <memory>
#include "Framework/Framework/SceneGraph.h"

namespace PhysIKA {
class AppBase
{
public:
    AppBase(void){};
    ~AppBase(){};

    virtual void createWindow(int width, int height){};
    virtual void mainLoop() = 0;
};

}  // namespace PhysIKA
