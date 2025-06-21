const readline = require('readline');
const startGame = require('./src/game');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

console.log('選擇模式：');
console.log('1) 純文字探索');
console.log('2) 地圖模式');
rl.question('模式: ', (ans) => {
  rl.close();
  const requireMap = ans.trim() === '2';
  startGame(requireMap);
});
