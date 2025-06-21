# NeuroGraph Memory Engine

本專案示範以短期記憶(STM)與長期記憶(LTM)為核心的簡易記憶系統，
並搭配小腦模組實作動作修正。`TextMincreft` 目錄內提供文字版的
沙盒遊戲，可作為線上訓練環境。

## 主要模組
- **TextEncoder**：將文字轉為向量表示。
- **ShortTermMemory / GNNLongTermMemory**：記錄與統整狀態轉移，長期記憶改以 GNN 實作。
  - STM 內含 L/R/C 三類連結強度，可透過 `stm.visualize('out.png')` 產生圖像檢視。
- **ReadNet**：從 STM/LTM 擷取相關資訊。
- **ActionDecoder**：根據融合後的狀態產生行為 logits。
- **DecisionInterface**：綜合 STM 與 LTM 決策路徑。
- **Consolidator**：於睡眠階段將 STM 歸化至 LTM。
- **Cerebellum**：依據回饋調整動作參數的小腦模組。
- **GNNLongTermMemory**：以圖神經網路實作的可微分 LTM，可於離線階段進行梯度更新。

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
6. 長期記憶模組已更新為 `GNNLongTermMemory`，訓練腳本會將 STM 內容整合並以梯度方式學習。
7. 程式每十步會呼叫 `stm.visualize()` 輸出 `stm_step_<N>.png`，方便觀察記憶連結。

## 執行測試
```
pip install -r requirements.txt
python -m pytest -q
```

詳細的 TextMincreft 操作可參考 `TextMincreft/README.md`。

## 4.1 可視化檢查方案
為便於檢查 GNNLongTermMemory 的訓練狀況，提供 `ltm_visualizer.py` 腳本進行圖譜輸出。

1. **節點空間分佈**：利用 t-SNE 將節點嵌入 `Kᵢ` 降維並以 Matplotlib 繪圖，節點顏色代表型別。
2. **邊強度熱圖**：讀取 `Wᵢⱼ` 後以 NetworkX 及線條粗細顯示權重大小。
3. **路徑梯度分析**：透過 autograd 計算某條路徑對預測值的梯度貢獻，將結果著色顯示。

執行範例：
```bash
python -m memory_engine.ltm_visualizer ltm_snapshot.pkl --outdir viz
```
輸出圖檔將存於指定資料夾內，方便人工驗證剪枝與訓練結果。

