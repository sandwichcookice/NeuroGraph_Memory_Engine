const fs = require('fs');
const path = require('path');

let logFile = null;

function init() {
  const dir = path.join(__dirname, '..', 'logs');
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir);
  }
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  logFile = path.join(dir, `history-${timestamp}.json`);
  fs.writeFileSync(logFile, '[]', 'utf8');
}

function append(command, response, state) {
  if (!logFile) init();
  const data = JSON.parse(fs.readFileSync(logFile, 'utf8'));
  data.push({
    time: new Date().toISOString(),
    command,
    response,
    state,
  });
  fs.writeFileSync(logFile, JSON.stringify(data, null, 2), 'utf8');
}

function read(file = logFile) {
  if (!file || !fs.existsSync(file)) return [];
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

module.exports = {
  init,
  append,
  read,
  get logFile() {
    return logFile;
  },
};
