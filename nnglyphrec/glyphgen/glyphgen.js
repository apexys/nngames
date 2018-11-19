const canvas = require('canvas')


function genGlyph(id){
    let cv = canvas.createCanvas(512,512);
    let ctx = cv.getContext('2d');
    //From here on out copied directly from receiveDonut
    //Generate bit pattern
    var v = id | 0;
    var vShift = 0
	var vBitPattern = [];
    do{
        vBitPattern.push((v & (1 << vShift)) ? 1 : 0);
    } while(v >> ++vShift);
    var vSplitBits = [ 1, 4, 7 ]
    for(var vI=0; vI < vSplitBits.length; vI += 1){
        vBitPattern.splice(vSplitBits[vI]-1, 0, 0);
    }
    var vHammingBits = [ 1, 2, 4, 8 ];
    for(var vI=0; vI < vHammingBits.length; ++vI){
        var vHammingBit = 0;
        for( var vCurrent=vHammingBits[vI]-1; vCurrent < vBitPattern.length; vCurrent += 2*vHammingBits[vI] ){
            for( var vAdded = 0; vAdded < vHammingBits[vI]; ++vAdded ){
                vHammingBit = vHammingBit ^ vBitPattern[vCurrent+vAdded];
            }
        }
        vBitPattern[vHammingBits[vI]-1] = vHammingBit; 
    }
    //Draw
    var count = 5 * 3 + 1 + 4
    let cseg = (a) => 1.5 * Math.PI + ((a / count) * 2 * Math.PI);
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, 512, 512);
    ctx.beginPath();
    ctx.fillStyle = "black";
    ctx.moveTo(256, 256);
    ctx.arc(256, 256, 256, cseg(0), cseg(4));
    ctx.closePath();
    ctx.fill();
    
    var vStart = 5;
    for(var vI = 0; vI < vBitPattern.length; ++vI){
        if( vBitPattern[vI] ){
            ctx.beginPath()
            ctx.moveTo(256, 256);
            ctx.arc(256, 256, 257, cseg(vStart+vI), cseg(vStart+vI+1));
            ctx.closePath();
            ctx.fill();
        }
    }
    ctx.beginPath();
    ctx.fillStyle = "white";
    ctx.arc(256, 256, 256 / 4 * 3, 0, 2 * Math.PI);
    ctx.closePath();
    ctx.fill();
    ctx.beginPath();
    ctx.fillStyle = "black";
    ctx.arc(256, 256, 256 / 2, 0, 2 * Math.PI);
    ctx.closePath();
    ctx.fill();
    ctx.beginPath();
    ctx.fillStyle = "white";
    ctx.arc(256, 256, 256 / 4, 0, 2 * Math.PI);
    ctx.closePath();
    ctx.fill();
    ctx.font = "64px sans-serif";
    ctx.fillStyle = "black";
    ctx.fillText(v, 12, 500);
    ctx.beginPath()
    ctx.moveTo(512 - 32, 0);
    ctx.lineTo(512, 64);
    ctx.lineTo(512 - 64, 64);
    ctx.closePath();
    ctx.fill();
    return cv;
}

var fs = require('fs');

if(fs.existsSync("glyphs")){
    require('rimraf').sync("glyphs");
}

fs.mkdirSync("glyphs");

for(var i = 0; i < 32; i++){
    console.log("Saving glyph " + i);
    var out = fs.createWriteStream('glyphs/' + i + ".png");
    genGlyph(i).createPNGStream().pipe(out);
}