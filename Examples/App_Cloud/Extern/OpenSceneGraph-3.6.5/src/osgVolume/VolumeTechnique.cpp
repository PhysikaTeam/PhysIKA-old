/* -*-c++-*- OpenSceneGraph - Copyright (C) 1998-2009 Robert Osfield
 *
 * This library is open source and may be redistributed and/or modified under
 * the terms of the OpenSceneGraph Public License (OSGPL) version 0.0 or
 * (at your option) any later version.  The full license is in LICENSE file
 * included with this distribution, and on the openscenegraph.org website.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * OpenSceneGraph Public License for more details.
*/

#include <osgVolume/VolumeTechnique>
#include <osgVolume/VolumeTile>

using namespace osgVolume;

VolumeTechnique::VolumeTechnique():
    _volumeTile(0)
{
    setThreadSafeRefUnref(true);
}

VolumeTechnique::VolumeTechnique(const VolumeTechnique& rhs,const osg::CopyOp& copyop):
    osg::Object(rhs,copyop),
    _volumeTile(0)
{
}

VolumeTechnique::~VolumeTechnique()
{
}

void VolumeTechnique::init()
{
    OSG_NOTICE<<className()<<"::initialize(..) not implemented yet"<<std::endl;
}

void VolumeTechnique::update(osgUtil::UpdateVisitor* uv)
{
    OSG_NOTICE<<className()<<"::update(..) not implemented yet"<<std::endl;
    if (_volumeTile) _volumeTile->osg::Group::traverse(*uv);
}

void VolumeTechnique::cull(osgUtil::CullVisitor* cv)
{
    OSG_NOTICE<<className()<<"::cull(..) not implemented yet"<<std::endl;
    if (_volumeTile) _volumeTile->osg::Group::traverse(*cv);
}

void VolumeTechnique::cleanSceneGraph()
{
    OSG_NOTICE<<className()<<"::cleanSceneGraph(..) not implemented yet"<<std::endl;
}

void VolumeTechnique::traverse(osg::NodeVisitor& nv)
{
    if (!_volumeTile) return;

    // if app traversal update the frame count.
    if (nv.getVisitorType()==osg::NodeVisitor::UPDATE_VISITOR)
    {
        if (_volumeTile->getDirty()) _volumeTile->init();

        osgUtil::UpdateVisitor* uv = nv.asUpdateVisitor();
        if (uv)
        {
            update(uv);
            return;
        }

    }
    else if (nv.getVisitorType()==osg::NodeVisitor::CULL_VISITOR)
    {
        osgUtil::CullVisitor* cv = nv.asCullVisitor();
        if (cv)
        {
            cull(cv);
            return;
        }
    }

    if (_volumeTile->getDirty()) _volumeTile->init();

    // otherwise fallback to the Group::traverse()
    _volumeTile->osg::Group::traverse(nv);
}

bool VolumeTechnique::isMoving(osgUtil::CullVisitor* cv)
{
    bool moving = false;

    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);

    ModelViewMatrixMap::iterator itr = _modelViewMatrixMap.find(cv->getIdentifier());
    if (itr!=_modelViewMatrixMap.end())
    {
        osg::Matrix newModelViewMatrix = *(cv->getModelViewMatrix());
        osg::Matrix& previousModelViewMatrix = itr->second;
        moving = (newModelViewMatrix != previousModelViewMatrix);

        previousModelViewMatrix = newModelViewMatrix;
    }
    else
    {
        _modelViewMatrixMap[cv->getIdentifier()] = *(cv->getModelViewMatrix());
    }
    return moving;
}
