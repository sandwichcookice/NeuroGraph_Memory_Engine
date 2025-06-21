const env = require('../env');
const logger = require('../logger');
const guide = require('../guide');
const world = require('../world');
const fs = require('fs');
const path = require('path');

// Load recipes from data file
const items = JSON.parse(
  fs.readFileSync(
    path.join(__dirname, '..', '..', 'data', 'items.json'),
    'utf8'
  )
);
const recipes = Object.fromEntries(
  Object.entries(items)
    .filter(([, data]) => data.craft)
    .map(([name, data]) => [name, { ...data.craft, station: data.station }])
);

function createHandler(options = {}) {
  const { requireMap = false } = options;
  return function handleCommand(agent, goals, input) {
    const [cmd, param] = input.split(',');
    let response = '';

  const forbidWhileBag = ['探索', '破壞方塊', '前進', '後退', '左轉', '右轉', '向後轉'];
  if (agent.bagOpen && forbidWhileBag.includes(cmd)) {
    response = '請先關閉背包';
    logger.append(input, response, {
      inventory: agent.getInventory(),
      bagOpen: agent.bagOpen,
      currentBlock: env.peekBlock(),
      world: world.active() ? world.getState() : null,
    });
    return response;
  }

  if (cmd === '建立地圖') {
    const [w, h, name] = (param || '').split(/[, ]+/);
    const width = parseInt(w, 10) || 5;
    const height = parseInt(h, 10) || width;
    const nm = name || 'world';
    world.create(width, height, nm);
    response = `已建立地圖 ${nm}`;
  } else if (cmd === '載入地圖') {
    try {
      world.load(param || 'world');
      response = `已載入地圖 ${param || 'world'}`;
    } catch (e) {
      response = '世界不存在';
    }
  } else if (cmd === '前進') {
    if (world.active()) {
      response = world.moveForward() ? '前進' : '無法移動';
    } else {
      response = '未進入地圖';
    }
  } else if (cmd === '後退') {
    if (world.active()) {
      response = world.moveBackward() ? '後退' : '無法移動';
    } else {
      response = '未進入地圖';
    }
  } else if (cmd === '左轉') {
    if (world.active()) {
      world.turnLeft();
      response = '左轉';
    } else {
      response = '未進入地圖';
    }
  } else if (cmd === '右轉') {
    if (world.active()) {
      world.turnRight();
      response = '右轉';
    } else {
      response = '未進入地圖';
    }
  } else if (cmd === '向後轉') {
    if (world.active()) {
      world.turnAround();
      response = '向後轉';
    } else {
      response = '未進入地圖';
    }
  } else if (cmd === '探索') {
    if (world.active()) {
      env.setCurrentBlock(world.look());
    } else if (requireMap) {
      response = '未進入地圖';
    } else {
      env.setCurrentBlock(env.explore());
    }
    if (!response) {
      const item = env.peekBlock();
      response = item ? `發現 ${item}` : '什麼都沒有找到';
    }
  } else if (cmd === '破壞方塊') {
    if (param !== '手' && (agent.inventory[param] || 0) === 0) {
      response = '缺少工具';
    } else {
      const result = env.breakBlock(param);
      if (result.item) {
        agent.addItem(result.item, 1);
        if (world.active()) world.removeCurrentBlock();
        response = `成功破壞，獲得 ${result.item}*1`;
      } else {
        response = result.error || '沒有可以破壞的方塊';
      }
    }
  } else if (input === '打開背包') {
    response = agent.openBag();
  } else if (input === '關閉背包') {
    response = agent.closeBag();
  } else if (cmd === '搜尋物品') {
    response = agent.searchItem(param);
  } else if (cmd === '攻略') {
    if (!param) {
      response = '請輸入目標物品';
    } else {
      response = guide(param);
    }
  } else if (cmd === '合成') {
    const recipe = recipes[param];
    if (recipe) {
      if (recipe.station && (agent.inventory[recipe.station] || 0) === 0) {
        response = `需要${recipe.station}`;
        logger.append(input, response, {
          inventory: agent.getInventory(),
          bagOpen: agent.bagOpen,
          currentBlock: env.peekBlock(),
          world: world.active() ? world.getState() : null,
        });
        goals.check(agent);
        return response;
      }

      let ok = true;
      for (const [item, count] of Object.entries(recipe.needs)) {
        if ((agent.inventory[item] || 0) < count) {
          ok = false;
          break;
        }
      }
      if (ok) {
        for (const [item, count] of Object.entries(recipe.needs)) {
          agent.removeItem(item, count);
        }
        for (const [item, count] of Object.entries(recipe.gives)) {
          agent.addItem(item, count);
        }
        if (recipe.goal) goals.addWoodPlanks(recipe.goal);
        const gains = Object.entries(recipe.gives)
          .map(([i, c]) => `${i}*${c}`)
          .join('、');
        const costs = Object.entries(recipe.needs)
          .map(([i, c]) => `${i}*${c}`)
          .join('、');
        response = `合成成功，獲得 ${gains}，消耗 ${costs}`;
      } else {
        response = '材料不足';
      }
    } else {
      response = '未知合成';
    }
  } else if (cmd === '搜尋') {
    const logs = logger.read();
    const found = logs.filter((l) => l.response.includes(param));
    response = found.length
      ? `在紀錄中找到 ${param}*${found.length} 來源紀錄`
      : '在紀錄中未找到相關紀錄';
  } else {
    response = '未知指令';
  }

    logger.append(input, response, {
      inventory: agent.getInventory(),
      bagOpen: agent.bagOpen,
      currentBlock: env.peekBlock(),
      world: world.active() ? world.getState() : null,
    });
    goals.check(agent);
    return response;
  };
}

module.exports = createHandler;
