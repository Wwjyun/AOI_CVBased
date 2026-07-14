# VisionFlow AOI GPU 實作清單

## 行為契約

- [x] Recipe 可分別記錄「切小圖 GPU」、「每一個 detector GPU」、「GUI 預覽 GPU」。
- [x] 所有 GPU 選項預設關閉；舊 recipe 未包含 GPU 欄位時，行為必須與目前 CPU 版本完全相同。
- [x] 勾選 GPU 時只透過額外的 `visionflow_cuda.dll` 執行 CUDA 工作，不依賴 Python 版 OpenCV CUDA。
- [x] DLL、NVIDIA driver 或 CUDA device 不可用時，依 recipe 設定回退 CPU，並在 log、進度訊息及結果中留下原因。
- [x] 不可在 GPU 路徑執行的輪廓分析、Qt 合成、檔案 I/O 維持 CPU。

## Recipe 與驗證

- [x] 新增頂層 `gpu` 設定：`tiling`、`display`、`dll_path`、`fallback_to_cpu`。
- [x] 每個 `detectors.<id>` 新增 `use_gpu`，可各自切換。
- [x] RecipeManager 驗證 GPU 欄位型別，並保持舊 recipe 相容。
- [x] 所有內建 recipes 寫入明確的 GPU 預設值（全部 `false`）。
- [x] 檢測結果 JSON 記錄 requested/active/backend/fallback reason，方便追查實際跑 CPU 或 GPU。

## 額外 CUDA DLL

- [x] 建立穩定的 C ABI 與 `ctypes` bridge，集中處理 DLL 尋址、載入、錯誤碼與 thread safety。
- [x] DLL 提供 CUDA device/capability 查詢。
- [x] DLL 提供 BGR/RGB/gray 色彩轉換，供 detector 與 GUI 預覽使用。
- [x] DLL 提供 ROI crop，供 grid、contour、pattern-match 三種切小圖模式使用。
- [x] DLL 提供 resize、Gaussian blur、global/adaptive threshold、morphology 等 detector 前處理 primitive。
- [x] DLL 與 bridge 對輸入 dtype、channel、連續記憶體、尺寸及 stride 做防呆。
- [x] 提供 RTX 3090 (`sm_86`) build script，並讓 PyInstaller 收入 DLL。

## 切小圖 / Tiler

- [x] `create_tiler` 接受 GPU 執行器，但未勾選時完全沿用 NumPy crop。
- [x] Grid tile crop 可用 DLL。
- [x] Contour tile crop 可用 DLL；`findContours` 與形狀幾何分析維持 CPU。
- [x] Pattern-match tile crop 可用 DLL；現階段 template matching/NMS 維持 CPU，另列效能優化。
- [x] Anchored grid tile crop 可用 DLL；現階段 template matching 維持 CPU。

## Detectors

- [x] DetectorManager 把各 detector 的 `use_gpu` 與共享 DLL runtime 傳入 detector。
- [x] 401：blur、morphology、gray、adaptive threshold 可走 DLL；contour/rotated rect 維持 CPU。
- [x] 401-1：gray、resize、blur、adaptive threshold、morphology 可走 DLL；contour/circle 計算維持 CPU。
- [x] 401-2：gray、blur、adaptive threshold 可走 DLL；contour/white ratio reduction 先維持 CPU。
- [x] 900：gray、global/adaptive threshold 可走 DLL；contour 與框距量測維持 CPU。
- [x] GPU detector 失敗時不可產生半套結果；整個 detector 本次執行回退原 CPU 前處理。

## GUI

- [x] Recipe Designer 新增切小圖與 GUI 預覽 GPU 勾選框、DLL 路徑、CPU fallback 選項。
- [x] 每一個 detector 列新增 GPU 勾選框，只有啟用 detector 時才生效。
- [x] 載入 recipe 時還原所有 GPU 勾選狀態；另存 recipe 時完整保存。
- [x] GUI 預覽勾選時由 DLL 做 BGR→RGB；QImage/QPixmap、overlay 與文字合成仍由 Qt/CPU 顯示。
- [x] GUI 顯示目前 backend（CUDA DLL / CPU / fallback 原因），避免「有勾但實際沒跑 GPU」不透明。
- [x] Designer 的 tile preview 使用 recipe 的 tiling GPU 設定。

## 測試與部署

- [x] CPU compileall、GUI offscreen smoke、CLI synthetic smoke 全數通過。
- [x] GPU 選項開啟但 DLL 缺少時，fallback smoke 通過且結果可辨識 fallback。
- [x] 以 mock DLL 驗證 recipe 開關確實路由到 GPU bridge。
- [ ] 在 RTX 3090 主機編譯 `visionflow_cuda.dll`。
- [ ] 在 RTX 3090 比對 CPU/GPU tile、binary mask、detector 結果與 GUI 色彩一致。
- [ ] 在 RTX 3090 執行 warm-up 與 10/100 張 benchmark，記錄傳輸與各階段耗時。
- [ ] 打包版於無 NVIDIA GPU 與有 RTX 3090 的兩種環境都可啟動。

## 後續效能優化（不影響第一版切換功能）

- [ ] 將 pattern matching 移入 DLL。
- [ ] 將多個前處理 primitive 合併成單次 upload/download pipeline。
- [ ] 加入 batch ROI、CUDA stream、pinned memory 與 VRAM-aware batch size。
- [ ] 評估 connected components / reduction，減少 binary mask 回傳 CPU 的成本。
