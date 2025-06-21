class Goals {
  constructor() {
    this.tasks = {
      '\u6728\u982d\u6536\u96C6\u8005': false,
      '\u5408\u6210\u6728\u68D2': false,
      '\u6728\u6750\u91CF\u7522\u8005': false,
    };
    this.woodPlankCrafted = 0;
  }

  check(agent) {
    const inv = agent.getInventory();
    if (!this.tasks['\u6728\u982d\u6536\u96C6\u8005'] && (inv['\u6728\u982D'] || 0) >= 1) {
      this.tasks['\u6728\u982d\u6536\u96C6\u8005'] = true;
      console.log('\u4EFB\u52D9 \u6728\u982d\u6536\u96C6\u8005 \u5B8C\u6210');
    }
    if (!this.tasks['\u5408\u6210\u6728\u68D2'] && (inv['\u6728\u68D2'] || 0) >= 4) {
      this.tasks['\u5408\u6210\u6728\u68D2'] = true;
      console.log('\u4EFB\u52D9 \u5408\u6210\u6728\u68D2 \u5B8C\u6210');
    }
    if (!this.tasks['\u6728\u6750\u91CF\u7522\u8005'] && this.woodPlankCrafted >= 10) {
      this.tasks['\u6728\u6750\u91CF\u7522\u8005'] = true;
      console.log('\u4EFB\u52D9 \u6728\u6750\u91CF\u7522\u8005 \u5B8C\u6210');
    }
  }

  addWoodPlanks(count) {
    this.woodPlankCrafted += count;
  }
}

module.exports = Goals;
