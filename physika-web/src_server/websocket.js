const fsPromise = require('fs').promises;
const fs = require('fs');
const JSZip = require('jszip');
const path = require('path');

const WSServer = require('ws').Server;
const server = require('http').createServer();
const app = require('./app');

const wss = new WSServer({
    server: server,
});
server.on('request', app);

//用户文件路径
const userPath = path.join(__dirname, '../data/user_file');
//轮询次数，每次间隔1秒
const queryCountMax = 60;

const prefetchFileInfo = {
    'SPH': ['ParticleData', '.vti'],
    'Cloth': ['cloth', '.obj'],
    'ParticleEvolution':['cloud','.vti']
};

wss.on('connection', function connection(ws) {
    ws.on('message', function incoming(message) {
        ws.binaryType = 'arraybuffer';

        const mObj = JSON.parse(message);
        //console.log("mObj: ", mObj);
        let queryCount = 0;

        if (mObj.usePrefetch) {
            const fileInfo = prefetchFileInfo[mObj.simType];
            const fileDir = path.join(userPath, mObj.userID, mObj.uploadDate.toString(), 'sim_data/');
            const fileName = fileDir + fileInfo[0] + '_' + mObj.frameIndex + fileInfo[1];
            let queryFileName;
            if (!mObj.isEnd) {
                queryFileName = fileDir + fileInfo[0] + '_' + (mObj.frameIndex + 1) + fileInfo[1];
                //console.log(queryFileName);
            }
            else {
                queryFileName = fileName;
            }
            const queryFile = () => {
                if (fs.existsSync(queryFileName)) {
                    fsPromise.readFile(fileName)
                        .then(data => {
                            console.log('Data size: ', data.byteLength);
                            //对文件进行压缩
                            const zip = new JSZip();
                            zip.file(fileName, data);
                            return zip.generateAsync({
                                type: 'arraybuffer',
                                compression: "DEFLATE",
                                compressionOptions: {
                                    level: 6
                                }
                            });
                        })
                        .then(zipData => {
                            console.log('ZipData size: ', zipData.byteLength);
                            ws.send(zipData);
                        })
                        .catch(err => {
                            console.log('Error in readFile: ', err);
                            //出错如何处理？
                            ws.send([]);
                        });
                }
                else {
                    //console.log("File not ready!");
                    ++queryCount;
                    if (queryCount < queryCountMax) {
                        setTimeout(queryFile, 1000);
                    }
                    else {
                        console.log("Query file timed out!");
                        ws.send([]);
                    }
                }
            }
            queryFile();
        }
        else {
            const filePath = path.join(userPath, mObj.userID, mObj.uploadDate.toString(), 'sim_data', mObj.fileName);
            fsPromise.readFile(filePath)
                .then(data => {
                    console.log(data.buffer);
                    //对文件进行压缩
                    const zip = new JSZip();
                    zip.file(mObj.fileName, data.buffer);
                    return zip.generateAsync({
                        type: 'arraybuffer',
                        compression: "DEFLATE",
                        compressionOptions: {
                            level: 6
                        }
                    });
                })
                .then(zipData => {
                    console.log('zipData', zipData);
                    ws.send(zipData);
                    //模拟网络延迟
                    // setTimeout(() => {
                    //     console.log('zipData', zipData);
                    //     ws.send(zipData);
                    // }, 7000);
                })
                .catch(err => {
                    console.log('Error in websocket! ', err);
                    //出错如何处理？
                    ws.send([]);
                });
        }


    });
});

//如何触发？
wss.on('close', function close() {
    console.log('Socket close.');
});

wss.on('error', function error(err) {
    console.log('Socket error.');
})

const port = process.env.PORT || 8888;
server.listen(port, () => {
    console.log('Listening at http://localhost:%s/index.html', port);
})