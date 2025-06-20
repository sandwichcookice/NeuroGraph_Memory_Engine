const {PythonShell} = require('python-shell');
const createHandler = require('./src/commands');
const Agent = require('./src/agent');
const Goals = require('./src/goals');
const logger = require('./src/logger');
const env = require('./src/env');
const path = require('path');

async function main() {
  const agent = new Agent();
  const goals = new Goals();
  logger.init();
  const handle = createHandler({requireMap: false});

  const pyshell = new PythonShell(
    path.join(__dirname, '..', 'memory_engine', 'trainer.py'),
    {
      pythonPath: path.join(__dirname, '..', 'macEnv', 'bin', 'python3'), // 修正 pythonPath 指向正確 venv
      env: { ...process.env, PYTHONPATH: path.join(__dirname, '..') }
    }
  );

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

  pyshell.on('error', (err) => {
    console.error('Error in Python shell:', err);
  });
  pyshell.on('close', () => {
    console.log('Python shell closed');
  });
  pyshell.on('stderr', (err) => {
    console.error('Python shell stderr:', err);
  });

}

main();
