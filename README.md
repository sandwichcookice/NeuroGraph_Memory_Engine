# NeuroGraph Memory Engine

本專案示範以短期記憶(STM)與長期記憶(LTM)為核心的簡易記憶系統，
並搭配小腦模組實作動作修正。`TextMincreft` 目錄內提供文字版的
沙盒遊戲，可作為線上訓練環境。

## 主要模組
- **TextEncoder**：將文字轉為向量表示。
- **ShortTermMemory / LongTermMemory**：記錄與統整狀態轉移。
- **ReadNet**：從 STM/LTM 擷取相關資訊。
- **ActionDecoder**：根據融合後的狀態產生行為 logits。
- **DecisionInterface**：綜合 STM 與 LTM 決策路徑。
- **Consolidator**：於睡眠階段將 STM 歸化至 LTM。
- **Cerebellum**：依據回饋調整動作參數的小腦模組。

### 小腦子系統
- **Motor Generator**：根據既有記憶產生初步控制參數。
- **Error Comparator**：比較預期與實際執行結果計算誤差。
- **Motor Corrector**：利用誤差及動量微調參數。
- **Motor Memory**：保存各動作的最佳化紀錄供下次引用。

## 訓練流程概觀
1. 由 `TextMincreft/train.js` 啟動訓練，中控腳本透過
   `python-shell` 與 `memory_engine/trainer.py` 互動。
2. Python 端根據目前記憶與小腦狀態選擇指令，Node 端執行後將
   包含背包資訊與回應的 JSON 傳回。
3. 記憶系統於每回合更新 STM，並在固定步數後透過 Consolidator
   進入「睡眠」整合階段，將 STM 歸化進 LTM。
4. 小腦會依據回饋調整各指令的權重，使探索策略逐步收斂至可
   成功取得鑽石鎬的流程。
5. LTM 會定期寫入 `ltm_snapshot.pkl`，以便下次訓練可延續先前狀態。

## 執行測試
```
pip install -r requirements.txt
python -m pytest -q
```

詳細的 TextMincreft 操作可參考 `TextMincreft/README.md`。
