# AOI GPU 加速 TODO（RTX 3090 24GB）

## 目標與硬體條件

- CPU：Intel Core i7-7700（4 核心 / 8 執行緒）
- GPU：NVIDIA GeForce RTX 3090（24GB VRAM）
- 目標：把大量像素運算、模板比對與未來 AI 推論移至 GPU，降低單張檢測時間並提高批次吞吐量。
- 原則：不是所有 OpenCV 操作搬到 GPU 都會更快。小圖、少量 ROI、輪廓迴圈、CSV/JSON 與磁碟存取仍適合 CPU；應讓影像一次上傳 GPU 後連續完成多個步驟，避免 CPU/GPU 反覆複製。

## P0：先量測，避免盲目改寫

- [ ] 在 `core/pipeline.py` 加入分段計時：讀圖、切圖、每個 detector、彙總、overlay、寫檔。
- [ ] 記錄影像尺寸、tile/ROI 數量、各 detector 耗時、GPU 上傳與下載耗時、顯存峰值。
- [ ] 建立固定測試集（小圖、大圖、少 ROI、多 ROI、PASS、NG），保存 CPU 基準值。
- [ ] 設定驗收門檻：單張與批次至少快 1.5 倍，檢測結果、bbox、PASS/NG 必須與 CPU 版一致或在明確容差內。
- [ ] 啟動時檢查 CUDA 是否可用；不可用或 GPU 記憶體不足時自動退回 CPU，不能讓產線中斷。

## P1：最值得優先搬到 GPU 的部分

### 1. Pattern Match / Search ROI 模板比對

位置：`core/tiler.py` 的 `cv2.matchTemplate`。

- [ ] 將全圖或大型 Search ROI 的模板比對改為 GPU 版本（OpenCV CUDA `cv2.cuda.createTemplateMatching`，或 CuPy/PyTorch 實作）。
- [ ] 灰階轉換、縮放、模板比對、局部最大值篩選盡量留在 GPU，最後只下載座標與分數。
- [ ] 模板圖片於 recipe 載入後常駐 GPU，不要每張圖重複上傳。
- [ ] 多張圖使用同一尺寸時重用 GPU buffer，減少配置顯存的成本。
- [ ] 保留 CPU `cv2.matchTemplate` 路徑，做結果一致性與效能比較。

中文說明：模板比對會在搜尋區域的每個位置做大量重複計算，是目前最適合 RTX 3090 平行處理的傳統 CV 工作。搜尋區越大、模板越多，效益通常越明顯。

### 2. 前處理：色彩轉換、模糊、二值化、形態學

位置：`core/tiler.py`、`detectors/detector_401.py`、`detector_401_1.py`、`detector_401_2.py`、`detector_900.py`。

- [ ] 建立共用 `GpuImageContext`，讓原圖、灰階圖、模糊圖與 binary mask 在同一次檢測中共用。
- [ ] 評估改用 GPU 的 `cvtColor`、`resize`、`GaussianBlur`、固定 threshold、dilate、erode、morphology。
- [ ] 相同 tile 被多個 detector 使用時，只做一次灰階與常用前處理，不要每個 detector 重算。
- [ ] Adaptive Threshold 若所用 OpenCV CUDA 版本沒有直接支援，評估以 CuPy/PyTorch kernel 實作；未達加速門檻則保留 CPU。
- [ ] 把整張圖或一批 ROI 一次送入 GPU，避免逐個小 tile 上傳。

中文說明：這些是逐像素運算，GPU 很擅長；但單一小 ROI 的傳輸時間可能比運算時間更久。因此要採用「整圖上傳、連續運算、最後下載結果」或「多 ROI 批次處理」。

### 3. 多 ROI / 多 Tile 批次處理

位置：`core/pipeline.py` 目前逐 tile、逐 detector 的雙層迴圈。

- [ ] 將同尺寸 ROI 組成 batch，再交給 GPU detector 一次運算。
- [ ] 重新設計 detector 介面，新增 `run_batch(images)`，CPU 版可保留預設逐張行為。
- [ ] 以 CUDA stream 實作上傳、運算、下載重疊，減少 GPU 等待。
- [ ] 根據 ROI 尺寸與 24GB VRAM 自動選 batch size，並保留安全顯存空間。
- [ ] GUI 單張檢測優先低延遲；資料夾批次模式優先高吞吐量，兩者使用不同 batch 策略。

中文說明：i7-7700 核心數較少，而 3090 有大量平行運算單元。把數百個 ROI 一個一個處理，無法發揮 GPU；批次化通常比單純把某個函式換成 CUDA 更重要。

## P2：可搬一部分，但要 CPU/GPU 混合

### 4. Detector 401 / 401-1 / 401-2

- [ ] 灰階、縮放、Gaussian Blur、二值化、morphology 放 GPU。
- [ ] `findContours`、`contourArea`、`arcLength`、`minAreaRect`、`minEnclosingCircle` 暫留 CPU。
- [ ] 僅下載 binary mask 或候選區域，不要下載所有中間影像。
- [ ] Detector 401-2 的白像素比例改成 GPU reduction（CuPy/PyTorch sum/count），只回傳統計值。

中文說明：OpenCV 的輪廓追蹤與幾何物件處理多半仍以 CPU API 為主。最實際的做法是 GPU 產生乾淨 mask，再由 CPU 處理少量輪廓，而不是強行全部 GPU 化。

### 5. Detector 900

- [ ] 外層固定 threshold 與內層 adaptive threshold 優先 GPU 化。
- [ ] 候選輪廓搜尋與外框/內框配對留在 CPU。
- [ ] 若候選數量很大，先在 GPU 做 connected-components 或區域統計以縮小候選集合。
- [ ] 檢查巢狀配對迴圈的候選數量；若 CPU 配對仍是瓶頸，再做空間索引或向量化，不必先寫 CUDA。

### 6. Overlay 與預覽縮放

- [ ] 大圖預覽的 resize、色彩轉換可評估 GPU 化。
- [ ] 大量矩形/圓形標註可評估 GPU 或先縮圖再畫；一般數量不多時維持 CPU。
- [ ] PNG 編碼、CSV/JSON、matrix CSV 與檔案寫入維持 CPU。

中文說明：產生報表通常受磁碟與 PNG 壓縮影響，GPU 不一定有明顯幫助。只有超大圖縮放或大量圖形繪製經量測確定是瓶頸時才值得改。

## P3：未來 AI Detector（3090 最大優勢）

- [ ] 導入 YOLO、RT-DETR 或分類/分割模型時使用 PyTorch CUDA 或 ONNX Runtime CUDA。
- [ ] 優先使用 FP16；完成精度驗證後再評估 TensorRT FP16/INT8。
- [ ] 模型只載入一次並常駐 3090，不要每張圖重新建立 session。
- [ ] 使用 batch inference、固定輸入尺寸與 pinned memory。
- [ ] 建立 warm-up，計時時排除第一次 CUDA context 與模型初始化成本。
- [ ] 監控顯存，避免 GUI、batch worker 各自載入一份大型模型造成 24GB 顯存浪費。

中文說明：RTX 3090 對深度學習推論的提升通常遠高於傳統輪廓運算。若未來加入 AI detector，這應成為 GPU 架構的核心，而非只依賴 OpenCV CUDA。

## 不建議搬到 GPU 的部分

- [ ] 保持 CPU：YAML recipe 讀取與驗證、PASS/NG 彙總、bbox 座標映射、JSON/CSV 組裝、檔案監控、GUI 控制邏輯、logging。
- [ ] 保持 CPU：少量 contour 幾何計算與很小的 ROI；除非 profiling 證明它們是瓶頸。
- [ ] 不要同時開太多 GPU batch worker；單張 3090 通常應由一個 GPU scheduler 統一排程。

中文說明：這些工作資料量小、分支多或受 I/O 限制，移到 GPU 會增加複雜度，通常不會更快。

## 建議技術路線

- [ ] 先確認 NVIDIA Driver 與 CUDA 相容性，記錄版本於診斷頁面。
- [ ] 注意：目前 `requirements.txt` 的 `opencv-python` 官方套件通常不含 CUDA；若選 OpenCV CUDA，需要自行編譯含 CUDA 的 OpenCV，或改用可信且版本鎖定的 CUDA build。
- [ ] 原型階段優先比較三條路：OpenCV CUDA、CuPy、PyTorch；以實測速度、部署難度及 PyInstaller 相容性決定。
- [ ] 若未來以 AI 為主，優先採 PyTorch/ONNX Runtime/TensorRT；若以模板比對及 morphology 為主，再評估 OpenCV CUDA/CuPy。
- [ ] GPU 功能使用 recipe 或設定開關：`auto`、`cpu`、`cuda`；預設 `auto`。
- [ ] 將 CUDA 依賴設為可選套件，CPU 安裝仍須能正常啟動與執行。

## 建議實作順序

- [ ] 第一階段：加入 profiler 與 CPU baseline，不改檢測結果。
- [ ] 第二階段：完成 CUDA 可用性檢查、GPU context、CPU fallback 與診斷資訊。
- [ ] 第三階段：先做 `matchTemplate` GPU 原型，這是最可能立即看見提升的項目。
- [ ] 第四階段：整合共用 GPU 前處理與 ROI batch，改寫 `run_batch()`。
- [ ] 第五階段：針對 401 系列與 900 detector 做 CPU/GPU 混合處理。
- [ ] 第六階段：加入 AI detector 的 FP16 batch inference。
- [ ] 第七階段：完成 GUI/CLI 設定、PyInstaller CUDA 打包、無 CUDA 電腦 fallback 測試。

## 每個階段的驗收項目

- [ ] CPU 與 GPU 對相同測試集產生相同 tile 數、PASS/NG、缺陷數量與座標（允許事先定義的浮點/像素容差）。
- [ ] 分別量測冷啟動、warm-up 後單張、10 張批次、100 張批次。
- [ ] 測試 GPU 不可用、CUDA 初始化失敗、顯存不足時能自動退回 CPU。
- [ ] 測試 GUI 仍保持回應，不在主執行緒等待 CUDA。
- [ ] 記錄平均值、P95、最大顯存、CPU 使用率及 GPU 使用率。
- [ ] 只有實測達到加速門檻的功能才正式取代 CPU 路徑，CPU 版持續作為 fallback 與正確性基準。

## 針對這台電腦的預設建議

- [ ] i7-7700 僅 4 核心，現有 batch worker 先維持 2～4 個；導入 GPU 後改為單一 GPU queue，不要讓 4 個 worker 無限制搶 GPU。
- [ ] RTX 3090 有 24GB VRAM，可先從 16、32、64 個同尺寸 ROI 的 batch size 實測，再依影像大小自動調整。
- [ ] 使用 FP16 AI 推論；傳統 threshold/mask 多使用 `uint8`，統計或模板分數視精度需求使用 FP16/FP32。
- [ ] 監控 GPU 溫度、功耗與長時間批次穩定性；3090 長時間滿載時要確保散熱與電源供應充足。
