//开启子线程调用python
const { spawn } = require('child_process');

function callPython(pathArray) {
    return new Promise((resolve, reject) => {
        let simConfigFileName;
        const process = spawn('python', [pathArray[0], pathArray[1], pathArray[2], pathArray[3], pathArray[4]]);
        //const process=spawn('D:/PhysIKA_merge_build_python/bin/Release/App_Cloth.exe');
        process.stdout.on('data', function (data) {
            //当脚本在控制台打印内容并返回收集输出数据的缓冲区时，将发出此事件
            //为了将缓冲区数据转换为可读形式，使用了toString()。
            simConfigFileName = data.toString();
        });
        process.on('close', (code) => {
            //当子进程的stdio流已关闭时，发出'close'事件，
            //此时再将所有数据传给浏览器
            console.log(`child process close all stdio with code ${code}`);
            console.log('simConfigFileName: ', simConfigFileName);
            resolve(simConfigFileName);
        });
        process.on('error', err => {
            console.log('Error in callPython!');
            reject(err);
        });

    });
}

module.exports = callPython;