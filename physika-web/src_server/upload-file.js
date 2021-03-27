const util = require('util');
//用于存储用户上传文件
const multer = require('multer');
const fsPromise = require('fs').promises;
const path = require('path');
const fs=require('fs');

//读取路径配置文件
const pathConfigFileName = path.join(__dirname, 'pathconfig.json');

const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        const extraInfo = req.body;
        console.log('extraInfo',extraInfo);
        fsPromise.readFile(pathConfigFileName, 'utf-8')
            .then(json => {
                const config = JSON.parse(json);
                const userPath = config.userPath;
                const userDir = path.join(userPath, extraInfo.userID);
                if (!fs.existsSync(userDir)) {
                    fs.mkdirSync(userDir);
                }
                const uploadFileDir = path.join(userDir, 'upload_file');
                if (!fs.existsSync(uploadFileDir)) {
                    fs.mkdirSync(uploadFileDir);
                }
                cb(null, uploadFileDir);
            })
            .catch(err => {
                cb(err);
            })
    },
    filename: function (req, file, cb) {
        const extraInfo = req.body;
        cb(null, extraInfo.uploadDate + '_' + file.originalname);
    }
});

async function uplaodFile(req, res) {
    try {
        const upload = util.promisify(multer({ storage: storage }).any());
        await upload(req, res);
        res.send('ok');
    } catch (err) {
        console.log('Error in multer!');
        res.sendStatus(500);
    }
}

module.exports = uplaodFile;