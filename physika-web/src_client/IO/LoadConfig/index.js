import { buildDataStructure } from '../BuildDataStructure';

function loadConfig(simType) {
    let reqBody = {
        "simType": simType
    };
    console.log(reqBody);

    //fetch是异步操作，
    //需要使用promise保证该函数能返回正确的data值
    return new Promise((resolve, reject) => {
        //-------从服务器获取初始配置
        fetch('/loadConfig', {
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
            console.log(err);
        });
    });
}

export {
    loadConfig as physikaLoadConfig
};

