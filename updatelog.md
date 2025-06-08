# Update Log

## [Unreleased]
- Initial implementation of core modules.
- 新增 TextEncoder 之 LRU 快取機制並新增對應測試。

- 修正 TextEncoder 輸出維度並調整 ReadNet 合併邏輯。

- 改寫 ReadNet 以向量化計算 STM 與 LTM 注意力。
- MemoryGraph 新增節點年齡避免新節點過早被移除。
- DecisionInterface 新增 plan_path 方法回傳完整路徑。
- 補充 LTM 動態注意力與相關單元測試。
- 新增小腦模組並整合至記憶系統。
- 新增 TextMincreft 訓練腳本 (train.js) 與 Python 端 trainer.py。
- 補充專案 README 介紹訓練流程。
- 重新設計 trainer.py 使其透過小腦參數進行探索式學習，
  並於睡眠階段將 STM 歸化至 LTM。
- 擴充小腦模組為完整版本，加入動量機制與記憶存取方法。
- 測試檔更新以符合新小腦介面。
