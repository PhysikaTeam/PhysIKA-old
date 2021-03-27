const express = require('express');
const path = require('path');
const fs = require('fs');
const fsPromise = require('fs').promises;
//引入body-parser用于解析post的body
const bodyParser = require('body-parser');
//加载xml
const xml2js = require('xml-js');
const app = express();
// create application/json parser
const jsonParser = bodyParser.json();
// create application/x-www-form-urlencoded parser
const urlencodedParser = bodyParser.urlencoded({ extended: false });

const callPython = require('./call-python');
const uplaodFile = require('./upload-file');

//读取路径配置文件
const pathConfigFileName = path.join(__dirname, 'pathconfig.json');

app.use(express.static(path.join(__dirname, '../dist')));

app.post('/loadConfig', jsonParser, function (req, res, next) {
    console.log('/loadConfig: \nreq.body: ', req.body);
    const simType = req.body.simType;
    fsPromise.readFile(pathConfigFileName, 'utf-8')
        .then(json => {
            //将json转化为js对象
            let config = JSON.parse(json);
            console.log(config);
            //通过key获取对象的value：若key是变量，则只能使用obj[key]；若key是唯一固定值，则可以使用obj['key']或obj.key。
            //获取初始化文件路径
            let initConfigFileName = config[simType].initConfigFileName;
            return fsPromise.readFile(initConfigFileName, 'utf-8');
        })
        .then(xml => {
            const options = { compact: true, ignoreComment: true };
            const result = xml2js.xml2js(xml, options);
            res.json(result);
            console.log('---------------');
        })
        .catch(err => {
            console.log(err);
            next(err);
        });
});

//上传配置的同时调用python
app.post('/uploadConfig', jsonParser, function (req, res, next) {
    console.log('/uploadConfig: \nreq.body: ', req.body);
    const extraInfo = req.body.extraInfo;
    const jsonObj = req.body.jsonObj;
    let uploadConfigFile = null;
    if (extraInfo.simType == 'SPH') {
        uploadConfigFile = JSON.stringify(jsonObj);
    }
    else {
        //将json转化为xml
        const options = { compact: true, ignoreComment: true, spaces: 4 };
        uploadConfigFile = xml2js.json2xml(jsonObj, options);
    }

    fsPromise.readFile(pathConfigFileName, 'utf-8')
        .then(json => {
            const config = JSON.parse(json);
            const userPath = config.userPath;
            const callPythonFileName = config[extraInfo.simType].callPythonFileName;
            //用户路径
            const userDir = path.join(userPath, extraInfo.userID);
            if (!fs.existsSync(userDir)) {
                fs.mkdirSync(userDir);
            }
            //上传时间路径
            const uploadDateDir = path.join(userDir, extraInfo.uploadDate.toString());
            if (!fs.existsSync(uploadDateDir)) {
                fs.mkdirSync(uploadDateDir);
            }
            //上传仿真配置文件
            let uploadConfigFileName = null;
            if (extraInfo.simType == 'SPH' || extraInfo.simType == 4) {
                uploadConfigFileName = path.join(uploadDateDir, 'upload_config_file.json');
            }
            else {
                uploadConfigFileName = path.join(uploadDateDir, 'upload_config_file.xml');
            }
            //仿真数据存储路径
            const simDataDir = path.join(uploadDateDir, 'sim_data');
            if (!fs.existsSync(simDataDir)) {
                fs.mkdirSync(simDataDir);
            }
            //上传文件目录
            const uploadFileDir = path.join(userDir, 'upload_file');
            return Promise.all([
                fsPromise.writeFile(uploadConfigFileName, uploadConfigFile),
                callPythonFileName,
                uploadConfigFileName,
                uploadFileDir,
                uploadDateDir,
                simDataDir
            ]);
        })
        .then(pathArray => {
            pathArray.shift();
            console.log(pathArray);
            console.log('当前运行路径：', process.cwd());
            return callPython(pathArray);
        })
        .then(simConfigFileName => {
            return fsPromise.readFile(simConfigFileName, 'utf-8');
        })
        .then(xml => {
            const options = { compact: true, ignoreComment: true };
            const result = xml2js.xml2js(xml, options);
            res.json(result);
            console.log('---------------');
        })
        .catch(err => {
            console.log(err);
            next(err);
        });
});

app.post('/uploadFile', uplaodFile);

module.exports = app;