import JSZip from 'jszip';
import vtkXMLImageDataReader from 'vtk.js/Sources/IO/XML/XMLImageDataReader';
import vtkVolume from 'vtk.js/Sources/Rendering/Core/Volume';
import vtkVolumeMapper from 'vtk.js/Sources/Rendering/Core/VolumeMapper';
import HttpDataAccessHelper from 'vtk.js/Sources/IO/Core/DataAccessHelper/HttpDataAccessHelper';

//vtkColorTransferFunction是RGB或HSV空间中的一种颜色映射
import vtkColorTransferFunction from 'vtk.js/Sources/Rendering/Core/ColorTransferFunction';
//vtkPiecewiseFunction用于定义分段函数映射
import vtkPiecewiseFunction from 'vtk.js/Sources/Common/DataModel/PiecewiseFunction';

import vtkBoundingBox from 'vtk.js/Sources/Common/DataModel/BoundingBox';

function initializeVti(vtiReader) {
    const source = vtiReader.getOutputData(0);
    const mapper = vtkVolumeMapper.newInstance();
    const actor = vtkVolume.newInstance();

    mapper.setInputData(source);
    actor.setMapper(mapper);

    /*
    //是否需要对体素进行一些通用的操作？
    const dataArray = source.getPointData().getScalars() || source.getPointData().getArrays()[0];
    const dataRange = dataArray.getRange();
    const lookupTable = vtkColorTransferFunction.newInstance();
    const piecewiseFunction = vtkPiecewiseFunction.newInstance();

    lookupTable.addRGBPoint(0, 85 / 255.0, 0, 0);
    lookupTable.addRGBPoint(95, 1.0, 1.0, 1.0);
    lookupTable.addRGBPoint(225, 0.66, 0.66, 0.5);
    lookupTable.addRGBPoint(255, 0.3, 1.0, 0.5);

    piecewiseFunction.addPoint(0.0, 0.0);
    piecewiseFunction.addPoint(255.0, 1.0);

    const sampleDistance = 0.7 * Math.sqrt(
        source.getSpacing()
            .map(v => v * v)
            .reduce((a, b) => a + b, 0)
    );
    mapper.setSampleDistance(sampleDistance);
    actor.getProperty().setRGBTransferFunction(0, lookupTable);
    actor.getProperty().setScalarOpacity(0, piecewiseFunction);
    actor.getProperty().setInterpolationTypeToLinear();
    //为了更好地查看体积，世界坐标中的绘制距离标量不透明度为1.0
    actor.getProperty().setScalarOpacityUnitDistance(
        0,
        vtkBoundingBox.getDiagonalLength(source.getBounds()) / Math.max(...source.getDimensions())
    );
    //表面边界，max应该在体积的平局梯度附近，或是平均值加上该梯度幅度一个标准偏差
    //（针对间距进行调整，这是世界坐标梯度，而不是像素梯度）
    //max的较好取值大小为：(dataRange[1] - dataRange[0]) * 0.05
    actor.getProperty().setGradientOpacityMinimumValue(0, 0);
    actor.getProperty().setGradientOpacityMaximumValue(0, (dataRange[1] - dataRange[0]) * 0.05);
    //使用基于渐变的阴影
    actor.getProperty().setShade(true);
    actor.getProperty().setUseGradientOpacity(0, true);
    //默认良好设置
    actor.getProperty().setGradientOpacityMinimumOpacity(0, 0.0);
    actor.getProperty().setGradientOpacityMaximumOpacity(0, 1.0);
    */

    actor.getProperty().setAmbient(0.2);
    actor.getProperty().setDiffuse(0.7);
    actor.getProperty().setSpecular(0.3);
    actor.getProperty().setSpecularPower(8.0);

    return { source, mapper, actor };
}

function loadVti(options) {
    const frameSeq = [];
    console.log(options);
    return new Promise((resolve, reject) => {
        if (options.file) {
            if (options.ext === 'vti') {
                const vtiReader = vtkXMLImageDataReader.newInstance();
                const fileReader = new FileReader();
                fileReader.onload = function onLoad(e) {
                    vtiReader.parseAsArrayBuffer(fileReader.result);
                    frameSeq.push(initializeVti(vtiReader));
                    resolve(frameSeq);
                }
                fileReader.readAsArrayBuffer(options.file);
            }
            else {
                //读取本地包含多个vti文件的zip？
            }
        }
        else if (options.fileURL) {
            if (options.ext === 'vti') {
                const vtiReader = vtkXMLImageDataReader.newInstance();
                HttpDataAccessHelper.fetchBinary(options.fileURL)
                    .then(res => {
                        vtiReader.parseAsArrayBuffer(res);
                        frameSeq.push(initializeVti(vtiReader));
                        resolve(frameSeq);
                    })
                    .catch(err => {
                        console.log("Failed to fetch .vti through url: ", err);
                    })
            }
            else {
                //读取url包含多个vti文件的zip？
                const zip = new JSZip();
                HttpDataAccessHelper.fetchBinary(options.fileURL)
                    .then(res => {
                        return zip.loadAsync(res);
                    })
                    .then(() => {
                        const fileContents = [];
                        zip.forEach((relativePath, zipEntry) => {
                            //正则表达式：两个斜杠（/）之间是模式；反斜杠（\）代表转义；$代表匹配结束；i为标志，表示不区分大小写搜索。
                            if (relativePath.match(/\.vti$/i)) {
                                //记录帧序，因为Promise.all不保证fileContents中文件的顺序，所以需要之后利用帧序重排序。
                                const frameIndex = relativePath.substring(relativePath.lastIndexOf('_') + 1, relativePath.lastIndexOf('.'));
                                const promise = new Promise((resolve, reject) => {
                                    zipEntry.async('arraybuffer')
                                        .then(res => {
                                            const vtiReader = new vtkXMLImageDataReader.newInstance();
                                            vtiReader.parseAsArrayBuffer(res);

                                            resolve({ frameIndex: frameIndex, vtiReader: vtiReader });
                                        })
                                        .catch(err => {
                                            reject(err);
                                        });
                                });
                                fileContents.push(promise);
                            }
                        });
                        return Promise.all(fileContents);
                    })
                    .then(res => {
                        //使用Array内置sort排序，按照frameIndex升序排列
                        res.sort(function (a, b) {
                            return (a.frameIndex - b.frameIndex);
                        });
                        res.forEach(item => {
                            frameSeq.push(initializeVti(item.vtiReader));
                        })
                        resolve(frameSeq);
                    })
                    .catch(err => {
                        console.log("Failed to fetch .zip through url: ", err);
                    });
            }
        }
        else {
            reject("不支持该文件格式！");
        }
    });
}

export {
    loadVti as physikaLoadVti
}