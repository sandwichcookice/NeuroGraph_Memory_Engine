const {PythonShell} = require('python-shell');
const createHandler = require('./src/commands');
const Agent = require('./src/agent');
const Goals = require('./src/goals');
const logger = require('./src/logger');
const env = require('./src/env');

async function main() {
  const agent = new Agent();
  const goals = new Goals();
  logger.init();
  const handle = createHandler({requireMap: false});

  const pyshell = new PythonShell('../memory_engine/trainer.py');

  pyshell.send('start');

  pyshell.on('message', (cmd) => {
    if (!cmd) return;
    if (cmd === 'exit') {
      pyshell.end(() => console.log('Training finished'));
      return;
    }
    const res = handle(agent, goals, cmd);
    console.log('> ' + cmd);
    console.log('-> ' + res);
    pyshell.send(JSON.stringify({inventory: agent.getInventory(), response: res}));
    goals.check(agent);
  });
}

main();
