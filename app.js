let url = 'https://raw.githubusercontent.com/CodingTrain/ColorClassifer-TensorFlow.js/master/colorData.json';
let data;
let r = 100;
let g = 100;
let b = 100;
let rReg = document.getElementById("colorR");
let gReg = document.getElementById("colorG");
let bReg = document.getElementById("colorB");
let body = document.querySelector("#canvas");
let h1 = document.getElementById("color");
let isdraw = true;
let model = tf.sequential();

let labelsToNum = [
    "red-ish",
    "green-ish",
    "blue-ish",
    "orange-ish",
    "yellow-ish",
    "pink-ish",
    "purple-ish",
    "brown-ish",
    "grey-ish"
];

let labelsToNumPl = [
    "czerwonawy",
    "zielonawy",
    "niebieskawy",
    "pomaranczowy",
    "żółtawy",
    "różowawy",
    "fioletowawy",
    "brązowawy",
    "szarawy"
];

setup();

function getPredictAsString(color) {
    let index = model.predict(tf.tensor2d([[rReg.value/255, gReg.value/255, bReg.value/255]])).argMax(1).dataSync();
    return labelsToNumPl[index];
}


function draw (){
    setInterval(()=>{
        if (isdraw == true){
        body.style.backgroundColor = `rgb(${rReg.value}, ${gReg.value}, ${bReg.value})`;
        h1.innerHTML = getPredictAsString();

    }}, 33);
    }



async function setup(){
    draw();
    await fetch(url).then((resp)=>{return resp.json()}).then((json)=>data = json["entries"]);

    

    let hidden = tf.layers.dense({
        units: 16,
        activation: "sigmoid",
        inputDim: 3
    });

    let output = tf.layers.dense({
        units: 9,
        activation: "softmax"
    });

    model.add(hidden);
    model.add(output);

    let optimizer = tf.train.sgd(0.3);

    model.compile({
        optimizer: optimizer,
        loss: "categoricalCrossentropy"
    });


    let colors = [];
    let labels = [];
    for (let dat of data){
        colors.push([dat.r/255, dat.g/255, dat.b/255]);
        labels.push(labelsToNum.indexOf(dat.label));
    }
    let xs = tf.tensor2d(colors);
    let tensorLabels = tf.tensor1d(labels, 'int32')
    let ys = tf.oneHot(tensorLabels, 9);
    tf.dispose(tensorLabels);


    model.fit(xs, ys, {
        epochs: 10,
        shuffle: true,
        validationSplit: 0.1,
        callbacks: {onEpochEnd: (epoch, logs) => console.log(logs.loss), onTrainEnd: ()=>{model.save('localstorage://my-model-1')} }
    }).then((res)=>{
        console.log(res);
    });


}

