class Agent {
  constructor() {
    this.inventory = {};
    this.bagOpen = false;
  }

  openBag() {
    this.bagOpen = true;
    return '\u80CC\u5305\u5DF2\u958B\u555F';
  }

  closeBag() {
    this.bagOpen = false;
    return '\u80CC\u5305\u5DF2\u95DC\u9589';
  }

  addItem(item, qty = 1) {
    if (!item) return;
    this.inventory[item] = (this.inventory[item] || 0) + qty;
  }

  removeItem(item, qty = 1) {
    if (this.inventory[item] >= qty) {
      this.inventory[item] -= qty;
      if (this.inventory[item] === 0) delete this.inventory[item];
      return true;
    }
    return false;
  }

  searchItem(item) {
    const count = this.inventory[item] || 0;
    if (count > 0) {
      return `\u80CC\u5305\u4E2D\u5305\u542B ${item}*${count}`;
    }
    return `\u80CC\u5305\u4E2D\u6C92\u6709 ${item}`;
  }

  getInventory() {
    return this.inventory;
  }
}

module.exports = Agent;
