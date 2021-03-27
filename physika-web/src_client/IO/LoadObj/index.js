import JSZip from 'jszip';
import vtkOBJReader from 'vtk.js/Sources/IO/Misc/OBJReader';
import vtkActor from 'vtk.js/Sources/Rendering/Core/Actor';
import vtkMapper from 'vtk.js/Sources/Rendering/Core/Mapper';
import HttpDataAccessHelper from 'vtk.js/Sources/IO/Core/DataAccessHelper/HttpDataAccessHelper';

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

function loadObj(options) {
    const frameSeq = [];
    console.log(options);
    return new Promise((resolve, reject) => {
        if (options.file) {
            if (options.ext === 'obj') {
                const objReader = vtkOBJReader.newInstance({ splitMode: 'usemtl' });
                const fileReader = new FileReader();
                //处理load事件，该事件在读取操作完成时触发
                fileReader.onload = function onLoad(e) {
                    //const objReader = vtkOBJReader.newInstance();
                    objReader.parseAsText(fileReader.result);
                    frameSeq.push(initializeObj(objReader));
                    resolve(frameSeq);
                };
                //开始读取指定的Blob中的内容
                fileReader.readAsText(options.file);
            }
            else {
                const zip = new JSZip();
                zip.loadAsync(options.file)
                    .then(() => {
                        //未完待续。。。
                    })
            }
        }
        else if (options.fileURL) {
            if (options.ext === 'obj') {
                //splitMode模式会将obj中的多个对象拆分存储，'usemtl'能否换成别的目前不清楚。。。
                const objReader = vtkOBJReader.newInstance({ splitMode: 'usemtl' });
                objReader.setUrl(options.fileURL)
                    .then(() => {
                        frameSeq.push(initializeObj(objReader));
                        resolve(frameSeq);
                    })
                    .catch(err => {
                        console.log("Failed to fetch .obj through url: ", err);
                    })
            }
            else {
                const zip = new JSZip();
                HttpDataAccessHelper.fetchBinary(options.fileURL)
                    .then(res => {
                        return zip.loadAsync(res);
                    })
                    .then(() => {
                        const fileContents = [];
                        zip.forEach((relativePath, zipEntry) => {
                            //正则表达式：两个斜杠（/）之间是模式；反斜杠（\）代表转义；$代表匹配结束；i为标志，表示不区分大小写搜索。
                            if (relativePath.match(/\.obj$/i)) {
                                //记录帧序，因为Promise.all不保证fileContents中文件的顺序，所以需要之后利用帧序重排序。
                                const frameIndex = relativePath.substring(relativePath.lastIndexOf('_') + 1, relativePath.lastIndexOf('.'));
                                const promise = new Promise((resolve, reject) => {
                                    zipEntry.async('string')
                                        .then(res => {
                                            const objReader = vtkOBJReader.newInstance({ splitMode: 'usemtl' });
                                            objReader.parseAsText(res);
                                            resolve({ frameIndex: frameIndex, objReader: objReader });
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
                            frameSeq.push(initializeObj(item.objReader));
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
    loadObj as physikaLoadObj
};