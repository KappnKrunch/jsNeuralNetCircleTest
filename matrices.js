



//matrices are described row by column
//matrix operations
class MOp{
  static matrix(m, n){ 
    creationKernel.setOutput([m, n]);
    return creationKernel(m, n);
  }
  
  static matrixFill(m, n, s){
    fillKernel.setOutput([m, n]);
    return fillKernel(m,n,s);
  }
  
  static matrixRand(m, n){
    randKernel.setOutput([m, n]);
    return randKernel(m, n);
  }
  
  static identity(n){
    identityKernel.setOutput([n, n])
    return identityKernel(n);
  }
  
  static add(a,b){
    const aM = a.length;
    const aN = a[0].length;
    const bM = b.length;
    const bN = b[0].length;
    
    if(aM != bM || aN != bN) print("error matrices of different sizes cannot be added");
    
    addKernel.setOutput([aN, bM]);
    return addKernel(a,b);
  }
  
  static sub(a,b){
    const aM = a.length;
    const aN = a[0].length;
    const bM = b.length;
    const bN = b[0].length;
    
    if(aM != bM || aN != bN) print("error matrices of different sizes cannot be subtracted");
    
    subKernel.setOutput([aN, bM])
    return subKernel(a,b);
  }
  
  static multiply(a,b){
    const aM = a.length;
    const aN = a[0].length;
    const bM = b.length;
    const bN = b[0].length;
    
    if(aN != bM) print("error matrices of incorrect sizes cannot be multiplied");
    
    multKernel.setOutput([bN, aM]);
    return multKernel(a, b, aN);
  }  
  
  static scale(a, s){
    const aM = a.length;
    const aN = a[0].length;
    
    scaleKernel.setOutput([aN, aM]);
    return scaleKernel(a, s);
  }
  
  static hadamard(a, b){
    const aM = a.length;
    const aN = a[0].length;
    const bM = b.length;
    const bN = b[0].length;
    
    if(aM != bM || aN != bN) print("error matrices of different sizes for hadamard");

    hadamardKernel.setOutput([aN, bM]);
    return hadamardKernel(a, b);
  }
  
  static transpose(a){
    const aM = a.length;
    const aN = a[0].length;

    transKernel.setOutput([aM, aN]);
    return transKernel(a);
  }
}






//kernels on the gpu
const creationKernel = gpu.createKernel(function(m, n) {
    return 0;
  }).setDynamicOutput(true);

const fillKernel = gpu.createKernel(function(m, n, s) {
    return s;
  }).setDynamicOutput(true);

const identityKernel = gpu.createKernel(function(n) {
      return this.thread.x == this.thread.y ? 1 : 0;
    }).setDynamicOutput(true);

const randKernel = gpu.createKernel(function(m, n) {
    return 2.0 * Math.random() - 1.0;
  }).setDynamicOutput(true);

const multKernel = gpu.createKernel(function(a, b, n) {
    let s = 0;
    for(let k = 0; k < n; k++) s += a[this.thread.y][k] * b[k][this.thread.x];
    return s;
  }).setDynamicOutput(true).setDynamicArguments(true);

const scaleKernel = gpu.createKernel(function(a, s) {
      return a[this.thread.y][this.thread.x] * s;
    }).setDynamicOutput(true).setDynamicArguments(true);

const addKernel = gpu.createKernel(function(a, b) {
      return a[this.thread.y][this.thread.x] + b[this.thread.y][this.thread.x];
    }).setDynamicOutput(true).setDynamicArguments(true);

const subKernel = gpu.createKernel(function(a, b) {
      return a[this.thread.y][this.thread.x] - b[this.thread.y][this.thread.x];
    }).setDynamicOutput(true).setDynamicArguments(true);

const transKernel = gpu.createKernel(function(a) {
      return a[this.thread.x][this.thread.y];
    }).setDynamicOutput(true).setDynamicArguments(true);

const hadamardKernel = gpu.createKernel(function(a, b) {
      return a[this.thread.y][this.thread.x] * b[this.thread.y][this.thread.x];
    }).setDynamicOutput(true).setDynamicArguments(true);