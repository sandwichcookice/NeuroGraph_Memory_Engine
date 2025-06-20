# Update Log

## [Unreleased]
- Initial implementation of core modules.
- 新增 TextEncoder 之 LRU 快取機制並新增對應測試。

- 修正 TextEncoder 輸出維度並調整 ReadNet 合併邏輯。

- 改寫 ReadNet 以向量化計算 STM 與 LTM 注意力。
- MemoryGraph 新增節點年齡避免新節點過早被移除。
- DecisionInterface 新增 plan_path 方法回傳完整路徑。
- 補充 LTM 動態注意力與相關單元測試。

### 2025-06-19
- 調整架構，將各模組移入獨立資料夾。
- 新增日誌與錯誤處理提升穩定性。

### 2025-06-20
- 新增注意力調度單元並加入測試。
