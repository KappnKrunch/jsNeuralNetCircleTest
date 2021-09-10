class Environment{
  constructor(params){
    this.network = new NeuralNet(params);
    this.iter = 0;
  }
  
  
  baseLine(X){ //takes in [[x],[y]] puts out the correct [[a],[b],[c]]
    
    const x = X[0][0];
    const y = X[1][0];
    const eps1 = 0.1; // radius center
    const eps2 = eps1 + 0.3/2 //ring
    const diffCX = 0.5 - x;
    const diffCY = 0.5 - y;
    const d = sqrt(diffCX*diffCX + diffCY*diffCY);
    
    //observation and answer
    /*
    const i = int(4 * (x * 400 + 1600 * y));
    const ans = [[baseImage.pixels[i]/255], 
                 [baseImage.pixels[i + 1]/255], 
                 [baseImage.pixels[i + 2]/255]];
    */

    //return ans;
    return [[d > eps2], [d <= eps2 && d > eps1], [d <= eps1]];
  }
  
  
  train(lR){
    let Z = [[0], [0], [0]];
    let Q;
    let G;
    let A;
    
    //make guess with feedforward
      Q = [[random()], [random()]];
      G = this.network.feedForward(Q);
      A = this.baseLine(Q);

      //backpropagate
      Z = MOp.scale(MOp.hadamard(MOp.sub(A, G), this.network.dSoftPlus(G)), -2 * lR);
      this.network.backpropagateWith(Z);
  }
  
  draw(){
    print("training epoch: " + this.network.getEpoch());
    //printMat(this.network.feedForward([[0.5], [0.5]]));
    
    fill(220);
    stroke(220);
    rect(0,400, 400,15);
    
    fill(0);
    stroke(0);
    textAlign(LEFT, TOP);
    text("Gen: " + this.network.getEpoch(), 0, 402);
    
    displayedTime = this.network.getEpoch();
    
    
    for(let x = 0; x < 400; x+=tileSize){
      for(let y = 0; y < 400; y+=tileSize){
        
        const i = 4 * (x + (0.5 * tileSize) + 400 * (y + (0.5 * tileSize)));
        const o = this.network.feedForward([[x/400], [y/400]]); //get output of the net
        
        const p = color(255*max(0,o[0]), 255*max(0,o[1]), 255*max(0,o[2]));
        
        //const p = color(baseImage.pixels[i], baseImage.pixels[i + 1], baseImage.pixels[i + 2]);
        
        fill(p);
        stroke(p);

        rect(x+1, y+1, tileSize-1, tileSize-1);  
      }
    }
  }
}