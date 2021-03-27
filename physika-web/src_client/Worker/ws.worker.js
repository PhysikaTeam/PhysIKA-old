import registerWebworker from 'webworker-promise/lib/register';
let ws;

registerWebworker(async function (wwMessage, emit) {
    return new Promise((resolve, reject) => {

        if (wwMessage.init) {
            console.log('Creating socket');
            ws = new WebSocket('ws://localhost:8888/');
            ws.binaryType = 'arraybuffer';
            ws.onopen = function () {
                console.log('Socket open.');
            }
        }

        if(wwMessage.close){
            ws.close();
        }

        ws.onclose = function () {
            console.log('Socket close.');
        }

        ws.onerror = function (event) {
            console.error("Socket error.", event);
        };

        if (wwMessage.data) {
            console.log("Client socket message:", wwMessage.data);
            let wsMessage = JSON.stringify(wwMessage.data);
            switch (ws.readyState) {
                case ws.CONNECTING:
                    setTimeout(() => {
                        ws.send(wsMessage);
                    }, 1000);
                    break;
                case ws.OPEN:
                    ws.send(wsMessage);
                    break;     
                default:
                    reject(new Error('Socket closed!'));
                    break;
            }
        }

        ws.onmessage = function (wsMessage) {
            let arrayBuffer = wsMessage.data;
            console.log('Socket server message', arrayBuffer);
            resolve(new registerWebworker.TransferableResponse(arrayBuffer, [arrayBuffer]));
        };

    })
})
