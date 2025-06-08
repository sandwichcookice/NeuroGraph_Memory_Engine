const fs = require('fs');
const path = require('path');

// Load item data
const items = JSON.parse(
  fs.readFileSync(path.join(__dirname, '..', '..', 'data', 'items.json'), 'utf8')
);

const recipes = Object.fromEntries(
  Object.entries(items)
    .filter(([, d]) => d.craft)
    .map(([name, d]) => [name, { ...d.craft, station: d.station }])
);

// Allow some common aliases or typos for item names
const aliases = {
  '鑽石搞': '鑽石鎬',
  '鑽石稿': '鑽石鎬',
};

function planCraft(target, qty = 1, inventory = {}) {
  target = aliases[target] || target;
  const steps = [];

  function ensure(item) {
    if ((inventory[item] || 0) > 0) return;
    helper(item, 1);
    inventory[item] = (inventory[item] || 0) + 1;
  }

  function helper(item, need) {
    const have = inventory[item] || 0;
    if (have >= need) {
      inventory[item] -= need;
      return;
    }
    const needToProduce = need - have;

    const recipe = recipes[item];
    if (!recipe) {
      const info = items[item] || {};
      if (info.breakTool && info.breakTool !== '手') {
        ensure(info.breakTool);
      }
      steps.push(`獲取,${item}*${needToProduce}`);
      inventory[item] = have + needToProduce - need;
      return;
    }

    const output = recipe.gives[item] || 1;
    const times = Math.ceil(needToProduce / output);

    if (recipe.station && (inventory[recipe.station] || 0) === 0) {
      ensure(recipe.station);
    }

    for (const [ing, count] of Object.entries(recipe.needs)) {
      helper(ing, count * times);
    }

    const produced = output * times;
    steps.push(`合成,${item}*${produced}`);
    inventory[item] = have + produced - need;
  }

  helper(target, qty);
  return steps.join(' > ');
}

module.exports = planCraft;
