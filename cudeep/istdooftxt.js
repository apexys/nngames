var result = "";
for(let i = 0; i < 10000; i++){
	result += `demodata/pictures/${i}.png ${i}\n`;
}

require('fs').writeFileSync("pictures2.txt", result, 'utf-8');