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
- 新增 LTM 快照機制並整合記憶圖規劃功能。
- 重新設計 STM 依據 L/R/C 強度並加入視覺化方法。
- 調整 trainer.py 以傳入獎勵並記錄原始文字。
- 測試新增 STM 視覺化案例與安裝 matplotlib。
- trainer.py 週期性輸出 STM 圖像以協助驗證。
- 新增 GNNLongTermMemory 模組與測試，更新 README 介紹。
- trainer.py 改用 GNNLongTermMemory，整合 Consolidator 與測試覆蓋。
- 新增 `ltm_visualizer.py` 提供 t-SNE 與邊權重熱圖等可視化功能。
- README 增補 4.1 可視化檢查方案說明。
- requirements.txt 加入 scikit-learn 以支援降維。
- 改進 GNNLongTermMemory.train_offline 以完整損失計算與正則化支援。
