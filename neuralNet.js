class NeuralNet{
  constructor(layersLayout){
    this.layersLayout = layersLayout; //list of ints
    this.lC = this.layersLayout.length;
    this.wBC = this.layersLayout.length-1;
    this.layers = new Array(this.lC); //this is a list of matrices
    this.weightsBiases = new Array(this.lC-1); //this is a list of matrices
    this.epoch = 0;
    
    if(layersLayout.length < 2) print("improper neural net layout");
    
    for(let i = 0; i < this.lC-1; i++){
      let m = this.layersLayout[i] + 1;
      let n = this.layersLayout[i+1];
      
      this.weightsBiases[i] = MOp.matrixRand(m, n);
    }
  }
  
  
  //returns the networks output
  feedForward(a){
    this.layers[0] = a;
    
    for(let i = 1; i < this.lC; i++){
      this.layers[i-1].push([1]);
      
      const Z = MOp.multiply(this.weightsBiases[i-1], this.layers[i-1]);
      
      //this.layers[i] = this.relu(Z);
      this.layers[i] = this.softPlus(Z);
    
      this.layers[i-1].pop();
    }
    
    return this.layers[this.lC - 1];
  }
  
  backpropagateFrom(b, lR){
    //derivative of error function
    let Z = MOp.scale(
      MOp.hadamard(
        MOp.sub(b, this.layers[this.lC-1]), 
        this.dRelu(this.layers[this.lC-1])
      ), -2 * lR); 
    
    this.backpropagateWith(Z);
  }
  
  backpropagateWith(Z){
    let gradient;
    let ZZ;
    
    for(let i = this.lC-1; i >= 1; i--){
      this.layers[i-1].push([1]);
      
      gradient = MOp.multiply(Z, MOp.transpose(this.layers[i-1]));
      
      this.layers[i-1].pop();
      
      ZZ = MOp.multiply(MOp.transpose(this.weightsBiases[i-1]), Z);
      ZZ.pop(); //almost the whole reason for making this
      
      //Z = MOp.hadamard(ZZ, this.dRelu(this.layers[i-1]));
      Z = MOp.hadamard(ZZ, this.dSoftPlus(this.layers[i-1]));
      
      this.weightsBiases[i-1] = MOp.sub(this.weightsBiases[i-1], gradient);
    }
    
    this.epoch++; //neural net has been updated
  }
  
  
  //accessors
  getLayer(n){
    return this.layers[n];
  }
  
  getWeight(n){
    return this.weightsBiases[n];
  }
  
  getEpoch(){
    return this.epoch;
  }
  
  
  //activation functions
  relu(a){
    const aM = a.length;
    const aN = a[0].length;
    
    reluKernel.setOutput([aN, aM])
    return reluKernel(a);
  }
  
  softPlus(a){
    const aM = a.length;
    const aN = a[0].length;
    
    softPlusKernel.setOutput([aN, aM])
    return softPlusKernel(a);
  }
  
  
  //activation function derivatives
  dRelu(a){
    const aM = a.length;
    const aN = a[0].length;
    
    dReluKernel.setOutput([aN, aM]);
    return dReluKernel(a);
  }
  
  dSoftPlus(a){
    const aM = a.length;
    const aN = a[0].length;
    
    dSoftPlusKernel.setOutput([aN, aM]);
    return dSoftPlusKernel(a);
  }
  
  saveNet(){
    var str = JSON.stringify(this.weightsBiases);
    save(str, "netGen"+this.getEpoch()+".json");
  }
  
  loadNet(path){
    /*
    let result;
    let response = fetch(path).next(response => {
      console.log(response);
    }); // resolves with response headers
    */
    //print(r);
  }
}











//kernels on gpu
const reluKernel = gpu.createKernel(function(a) {
  return Math.max(a[this.thread.y][this.thread.x], 0);
}).setDynamicOutput(true).setDynamicArguments(true);

const dReluKernel = gpu.createKernel(function(a) {
  return a[this.thread.y][this.thread.x] > 0 ? 1 : 0;
}).setDynamicOutput(true).setDynamicArguments(true);

const softPlusKernel = gpu.createKernel(function(a) {
  return Math.log(1 + Math.exp(a[this.thread.y][this.thread.x]));
}).setDynamicOutput(true).setDynamicArguments(true);

const dSoftPlusKernel = gpu.createKernel(function(a) {
  return 1 / (1 + Math.exp(-(a[this.thread.y][this.thread.x])));
}).setDynamicOutput(true).setDynamicArguments(true);