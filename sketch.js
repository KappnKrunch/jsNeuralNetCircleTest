let tileSize = 40;
let env;
let time = 0;
let displayedTime;
let baseImage;

function preload(){
  baseImage = loadImage('assets/vanGogh.jpg');
}

function setup() {
  createCanvas(450, 450);
  
  env = new Environment([2,512,64,3]);
  
  baseImage.loadPixels();

  button = createButton('res up');
  button.position(400, 65);
  button.mousePressed(resUp);
  
  button = createButton('res down');
  button.position(400, 95);
  button.mousePressed(resDown);
  
  button = createButton('save image');
  button.position(400, 205);
  button.mousePressed(saveScreen);
  
  button = createButton('save net');
  button.position(400, 235);
  button.mousePressed(saveNet);
}                                                                                                                                                                     

function draw() { 
  if(time % 101 == 0){ 
    env.draw();
    
    print("Training...");
  }
  else{
    env.train(0.0001);
  }
  
  time++;
}


//debug printer function
function printMat(m){
  let s = "";
  for(let r = 0; r < m.length; r++){
    for(let c = 0; c < m[0].length; c++){
      s += m[r][c] + " ";
    }
    
    s += '\n';
  }
  
  print(s);
}


//sketch buttons
function resUp(){
  if(tileSize > 5) tileSize /= 2;
}

function resDown(){
  if(tileSize < 80) tileSize *= 2;
}

function saveScreen(){
  saveCanvas("neural_net_gen" + displayedTime, 'png');
}

function saveNet(){
  neuralNet.saveNet();
}