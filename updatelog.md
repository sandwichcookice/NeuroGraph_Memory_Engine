# Update Log

## [Unreleased]
- Initial implementation of core modules.
- 新增 TextEncoder 之 LRU 快取機制並新增對應測試。

- 修正 TextEncoder 輸出維度並調整 ReadNet 合併邏輯。

- 改寫 ReadNet 以向量化計算 STM 與 LTM 注意力。
- MemoryGraph 新增節點年齡避免新節點過早被移除。
- DecisionInterface 新增 plan_path 方法回傳完整路徑。
- 補充 LTM 動態注意力與相關單元測試。
