const readline = require('readline');
const Agent = require('./agent');
const Goals = require('./goals');
const createHandler = require('./commands');
const logger = require('./logger');

function start(requireMap) {
  const agent = new Agent();
  const goals = new Goals();
  logger.init();
  const handleCommand = createHandler({ requireMap });

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: '> '
  });

  console.log('TextMincreft \u555F\u52D5\uFF0C\u8F38\u5165\u6307\u4EE4\uFF1A');
  rl.prompt();

  rl.on('line', (line) => {
    const command = line.trim();
    const response = handleCommand(agent, goals, command);
    console.log('\u2192 ' + response);
    rl.prompt();
  }).on('close', () => {
    console.log('\u904A\u6232\u7D50\u675F');
    process.exit(0);
  });
}

module.exports = start;
