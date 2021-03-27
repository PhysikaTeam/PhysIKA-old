import JSZip from 'jszip';
import vtkOBJReader from 'vtk.js/Sources/IO/Misc/OBJReader';
import vtkActor from 'vtk.js/Sources/Rendering/Core/Actor';
import vtkMapper from 'vtk.js/Sources/Rendering/Core/Mapper';
import vtkCubeSource from 'vtk.js/Sources/Filters/Sources/CubeSource';

function initializeObj(objReader) {
    let curFrame = {};
    const nbOutputs = objReader.getNumberOfOutputPorts();
    for (let i = 0; i < nbOutputs; i++) {
        const source = objReader.getOutputData(i);
        const mapper = vtkMapper.newInstance();
        const actor = vtkActor.newInstance();
        let name = source.get('name').name;
        if (!name) {
            name = i;
        }

        mapper.setInputData(source);
        actor.setMapper(mapper);

        curFrame[name] = { source, mapper, actor };
    }
    return curFrame;
}

function initObj(arrayBuffer, ext) {
    return new Promise((resolve, reject) => {
        const objReader = vtkOBJReader.newInstance({ splitMode: 'usemtl' });
        if (ext === 'obj') {
            objReader.parseAsText(arrayBuffer);
            resolve(initializeObj(objReader));
        }
        else if (ext === 'zip') {
            const zip = new JSZip();
            zip.loadAsync(arrayBuffer)
                .then(() => {
                    zip.forEach((relativePath, zipEntry) => {
                        //正则表达式：两个斜杠（/）之间是模式；反斜杠（\）代表转义；$代表匹配结束；i为标志，表示不区分大小写搜索。
                        if (relativePath.match(/\.obj$/i)) {
                            zipEntry.async('string')
                                .then(res => {
                                    objReader.parseAsText(res);
                                    resolve(initializeObj(objReader));
                                })
                                .catch(err => {
                                    reject(err);
                                });
                        }
                        else {
                            return Promise.reject('压缩文件中不是obj文件！')
                        }
                    });
                })
                .catch(err => {
                    console.log("Failed to init obj: ", err);
                })
        }
        else if (ext === 'vtkCube') {
            const source = vtkCubeSource.newInstance();
            const actor = vtkActor.newInstance();
            const mapper = vtkMapper.newInstance();
            actor.setMapper(mapper);
            mapper.setInputConnection(source.getOutputPort());
            resolve([{ source, mapper, actor }]);
        }
        else {
            reject('数据格式错误！')
        }
    });
}

export {
    initObj as physikaInitObj
};