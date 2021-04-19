#include <osgManipulator/TrackballDragger>
#include <osgDB/ObjectWrapper>
#include <osgDB/InputStream>
#include <osgDB/OutputStream>

REGISTER_OBJECT_WRAPPER( osgManipulator_TrackballDragger,
                         new osgManipulator::TrackballDragger,
                         osgManipulator::TrackballDragger,
                         "osg::Object osg::Node osg::Transform osg::MatrixTransform osgManipulator::Dragger "
                         "osgManipulator::TrackballDragger" )  // No need to contain CompositeDragger here
{
    ADD_FLOAT_SERIALIZER(AxisLineWidth, 2.0f);
    ADD_FLOAT_SERIALIZER(PickCylinderHeight, 0.15f);
}
