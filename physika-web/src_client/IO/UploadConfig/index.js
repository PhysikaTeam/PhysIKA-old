import { deepCopy } from '../../Common'
import { buildDataStructure } from '../BuildDataStructure'

const buildJson = (father, children) => children.forEach(item => {
    //删除增加结点的结点
    if (item.tag === 'AddNode') {
        return;
    }
    if (!father.hasOwnProperty(item.tag)) {
        father[item.tag] = [];
    }
    father[item.tag].push(item);
    if (item.hasOwnProperty('children')) {
        buildJson(item, item.children);
        delete item.children;
    }

    //2021.3.20 若上传文件为xml，则需要在这里将_text值改为字符串！
    switch (item._attributes.class) {
        case 'File':
            const fileInfo = item._text[0];
            item._text = fileInfo.uploadDate + '_' + fileInfo.name;
            break;
        case 'Vector2u':
            item._text = item._text[0] + ' ' + item._text[1];
            break;
        default:
            break;
    }

    delete item.deletable
    delete item.key;
    delete item.tag;
});

function buildSPHJson(obj, jsonObj) {
    console.log('buildSPHJson: ', obj);
    Object.keys(obj.Scene[0]).forEach(key => {
        switch (key) {
            case 'Configuration':
                const configuration = obj.Scene[0].Configuration[0];
                delete configuration._attributes;
                const obj_0 = {};
                Object.keys(configuration).forEach(key => {
                    obj_0[key] = configuration[key][0]._text;
                });
                jsonObj.Configuration = obj_0;
                break;
            case 'FluidBlocks':
                const fluidBlocks = obj.Scene[0].FluidBlocks[0];
                delete fluidBlocks._attributes;
                const array_1 = [];
                Object.keys(fluidBlocks).forEach(key => {
                    const obj_1 = {};
                    const fluidBlock_i = fluidBlocks[key][0];
                    delete fluidBlock_i._attributes;
                    Object.keys(fluidBlock_i).forEach(i => {
                        obj_1[i] = fluidBlock_i[i][0]._text;
                    })
                    array_1.push(obj_1);
                });
                jsonObj.FluidBlocks = array_1;
                break;
            case 'Materials':
                const materials = obj.Scene[0].Materials[0];
                delete materials._attributes;
                const array_2 = [];
                Object.keys(materials).forEach(key => {
                    const obj_2 = {};
                    const materials_i = materials[key][0];
                    delete materials_i._attributes;
                    Object.keys(materials_i).forEach(i => {
                        obj_2[i] = materials_i[i][0]._text;
                    })
                    array_2.push(obj_2);
                });
                jsonObj.Materials = array_2;
                break;
            case 'RigidBodies':
                const rigidBodies = obj.Scene[0].RigidBodies[0];
                delete rigidBodies._attributes;
                const array_3 = [];
                Object.keys(rigidBodies).forEach(key => {
                    const obj_3 = {};
                    const rigidBodies_i = rigidBodies[key][0];
                    delete rigidBodies_i._attributes;
                    Object.keys(rigidBodies_i).forEach(i => {
                        obj_3[i] = rigidBodies_i[i][0]._text;
                    })
                    array_3.push(obj_3);
                });
                jsonObj.RigidBodies = array_3;
                break;
            default:
                break;
        }
    });
}

//上传数据到服务器
function uploadConfig(data, extraInfo) {
    console.log("开始上传");
    //jsonObj为导出xml的json对象
    let jsonObj = {};
    let obj = {};
    //必须用deepcopy，因为在深拷贝中会将html标签省略掉！！
    //如果不去掉html，将会产生循环引用
    buildJson(obj, deepCopy(data));

    if (extraInfo.simType == 'SPH' || extraInfo.simType == 4) {
        buildSPHJson(obj, jsonObj);
    }
    else {
        jsonObj._declaration = {
            _attributes: {
                version: "1.0",
                encoding: "utf-8"
            },
            Scene: {}
        }
        jsonObj.Scene = obj.Scene;
    }

    console.log("jsonObj", jsonObj);
    //reqBody为传入后端的请求对象
    let reqBody = {
        extraInfo: extraInfo,
        jsonObj: jsonObj
    };

    //fetch是异步操作，
    //需要使用promise保证该函数能返回正确的data值
    return new Promise((resolve, reject) => {
        fetch('/uploadconfig', {
            method: 'POST',
            body: JSON.stringify(reqBody),
            headers: new Headers({
                'Content-Type': 'application/json'
            })
        }).then(res => {
            if (res.ok) {
                return res.json();
            }
            console.log("发生了值得注意的其他错误！");
            return Promise.reject(res);
        }).then(res => {
            console.log(res);
            resolve(buildDataStructure(res));
        }).catch(err => {
            reject(err);
        });
    });
}

export {
    uploadConfig as physikaUploadConfig
};