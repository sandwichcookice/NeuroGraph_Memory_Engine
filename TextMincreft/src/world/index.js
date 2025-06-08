const fs = require('fs');
const path = require('path');
const env = require('../env');

let world = null;
let currentFile = null;

function getOffset() {
  if (!world) return { ox: 0, oy: 0 };
  return {
    ox: Math.floor(world.width / 2),
    oy: Math.floor(world.height / 2),
  };
}

function ensureDir() {
  const dir = path.join(__dirname, '..', '..', 'worlds');
  if (!fs.existsSync(dir)) fs.mkdirSync(dir);
  return dir;
}

function create(width = 5, height = 5, name = 'world') {
  const dir = ensureDir();
  const map = [];
  for (let y = 0; y < height; y++) {
    const row = [];
    for (let x = 0; x < width; x++) {
      row.push(env.randomBlock());
    }
    map.push(row);
  }
  world = { width, height, map, x: 0, y: 0, dir: 0 };
  currentFile = path.join(dir, `${name}.json`);
  save();
  return world;
}

function load(name = 'world') {
  const file = path.join(ensureDir(), `${name}.json`);
  if (!fs.existsSync(file)) throw new Error('世界不存在');
  world = JSON.parse(fs.readFileSync(file, 'utf8'));
  currentFile = file;
  return world;
}

function save() {
  if (!currentFile || !world) return;
  fs.writeFileSync(currentFile, JSON.stringify(world, null, 2), 'utf8');
}

function active() {
  return world !== null;
}

function getState() {
  return world;
}

function moveForward() {
  if (!world) return false;
  const { ox, oy } = getOffset();
  if (world.dir === 0 && world.y + oy > 0) {
    world.y--;
  } else if (world.dir === 1 && world.x + ox < world.width - 1) {
    world.x++;
  } else if (world.dir === 2 && world.y + oy < world.height - 1) {
    world.y++;
  } else if (world.dir === 3 && world.x + ox > 0) {
    world.x--;
  } else {
    return false;
  }
  save();
  return true;
}

function moveBackward() {
  if (!world) return false;
  turnAround();
  const ok = moveForward();
  turnAround();
  return ok;
}

function turnLeft() {
  if (!world) return;
  world.dir = (world.dir + 3) % 4;
  save();
}

function turnRight() {
  if (!world) return;
  world.dir = (world.dir + 1) % 4;
  save();
}

function turnAround() {
  if (!world) return;
  world.dir = (world.dir + 2) % 4;
  save();
}

function look() {
  if (!world) return null;
  const { ox, oy } = getOffset();
  return world.map[world.y + oy][world.x + ox];
}

function removeCurrentBlock() {
  if (!world) return;
  const { ox, oy } = getOffset();
  world.map[world.y + oy][world.x + ox] = null;
  save();
}

module.exports = {
  create,
  load,
  save,
  active,
  getState,
  moveForward,
  moveBackward,
  turnLeft,
  turnRight,
  turnAround,
  look,
  removeCurrentBlock,
};
