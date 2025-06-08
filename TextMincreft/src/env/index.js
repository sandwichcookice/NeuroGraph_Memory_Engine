const fs = require('fs');
const path = require('path');

// Load item definitions from JSON. Items marked with
// "obtain" including "explore" are possible random drops.
const data = JSON.parse(
  fs.readFileSync(
    path.join(__dirname, '..', '..', 'data', 'items.json'),
    'utf8'
  )
);

const possibleDrops = [
  ...Object.keys(data).filter(
    (name) => data[name].obtain && data[name].obtain.includes('explore')
  ),
  null,
  null,
];

let currentBlock = null;

function explore() {
  currentBlock =
    possibleDrops[Math.floor(Math.random() * possibleDrops.length)];
  return currentBlock;
}

function randomBlock() {
  return possibleDrops[Math.floor(Math.random() * possibleDrops.length)];
}

function setCurrentBlock(block) {
  currentBlock = block;
}

function breakBlock(tool) {
  if (currentBlock === null) {
    return { error: '沒有可以破壞的方塊' };
  }

  const info = data[currentBlock] || {};
  if (info.breakTool) {
    const required = info.breakTool;
    const tiers = ['木鎬', '石鎬', '鐵鎬', '鑽石鎬'];

    if (required === '手') {
      if (tool !== '手') {
        return { error: `需要${required}` };
      }
    } else if (tiers.includes(required)) {
      const need = tiers.indexOf(required);
      const have = tiers.indexOf(tool);
      if (have < need) {
        return { error: `需要${required}` };
      }
    } else if (tool !== required) {
      return { error: `需要${required}` };
    }
  }

  const drop = currentBlock;
  currentBlock = null;
  return { item: drop };
}

function peekBlock() {
  return currentBlock;
}

module.exports = {
  explore,
  breakBlock,
  peekBlock,
  randomBlock,
  setCurrentBlock,
};
