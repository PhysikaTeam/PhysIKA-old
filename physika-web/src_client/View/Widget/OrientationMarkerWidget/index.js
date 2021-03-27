//坐标轴
import vtkAxesActor from 'vtk.js/Sources/Rendering/Core/AxesActor';
//旋转控制控件
import vtkOrientationMarkerWidget from 'vtk.js/Sources/Interaction/Widgets/OrientationMarkerWidget';

function getOrientationMarkerWidget(renderWindow) {
    const axesActor = vtkAxesActor.newInstance();
    const orientationMarkerWidget = vtkOrientationMarkerWidget.newInstance({
        actor: axesActor,
        interactor: renderWindow.getInteractor(),
    });
    orientationMarkerWidget.setViewportCorner(
        vtkOrientationMarkerWidget.Corners.BOTTOM_LEFT
    );
    //控制控件大小
    orientationMarkerWidget.setViewportSize(0.3);
    orientationMarkerWidget.setMinPixelSize(100);
    orientationMarkerWidget.setMaxPixelSize(300);

    return orientationMarkerWidget;
}

export {
    getOrientationMarkerWidget,
}