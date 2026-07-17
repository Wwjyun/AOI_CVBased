# VisionFlow AOI 統一開發清單

本文件是專案唯一的工作清單，涵蓋 CPU、GPU、CUDA、Detector、GUI、打包、CI 與實機驗收。完成程式修改時必須同步更新對應 checkbox；不得再建立分散的 CPU/GPU Todo 文件。

## 開發原則

- CPU 路徑是正確性基準，也是無 NVIDIA GPU、DLL 載入失敗、CUDA error 或顯存不足時的 fallback。
- GPU 最佳化不得改變 recipe 語意、PASS/NG、座標與輸出格式；允許差異必須先定義容差並加入測試。
- 不追求所有工作 GPU 化。YAML、彙總、少量 contour 幾何、GUI 控制、CSV/JSON、PNG 編碼與磁碟 I/O 預設留在 CPU。
- Detector 不得各自建立一套 CUDA workflow。Detector 只宣告 backend-neutral `PreprocessPlan`，由 CPU/CUDA executor 執行共用 operators。
- GPU 路徑應盡量一次 upload、連續執行多個 operators、最後只 download 必要 mask 或統計值。
- 新功能必須保持 OOP 邊界、CPU-only 可啟動、舊 DLL 相容與完整 detector CPU fallback。
- 只有 RTX 3090 實測通過數值等價、穩定性與端到端效能門檻的功能，才能預設啟用 GPU。

## 目前狀態摘要

- [x] CUDA DLL 已在 RTX 3090 編譯，並在另一台電腦確認可載入及顯示 CUDA active。
- [x] 已確認 CUDA active 不代表整條 AOI pipeline 都在 GPU；首次跨機測試端到端沒有加速。
- [x] 已有 CPU-only、缺 DLL fallback、GPU 呼叫統計及 detector 整體 CPU 重跑機制。
- [x] Gaussian 已改 separable kernels；Adaptive Mean 已改 64-bit integral image。
- [x] 已有 persistent context、grow-only buffers 與 401-2 fused preprocessing 原型。
- [x] 已建立通用 `PreprocessPlan`、typed operators、CPU/CUDA executors，401-2 已完成第一階段遷移。
- [ ] 目前開發機缺少可用的 `nvcc`/CMake，新增 CUDA 原始碼仍需在 RTX 3090 重新編譯與實測。
- [ ] 尚未完成固定 production 測試集、五個 recipes 全流程等價、長時間壓測與可信的 CPU/GPU benchmark。

## P0：正確性、CPU 基準與觀測能力

### Pipeline 與 profiler

- [x] 記錄 recipe setup、image load、tiling、各 detector、aggregation、reporting 與 end-to-end host wall time。
- [x] Reporter 分別記錄 overlay、NG tiles、CSV、matrix CSV 與 JSON 耗時。
- [x] 記錄 DLL load、同步呼叫、lock wait、估算 H2D/D2H bytes、round trips 與各 primitive 呼叫統計。
- [x] 加入 GUI 顯示、QImage/QPixmap 轉換與使用者實際等待時間計時。
- [ ] DLL 加入 CUDA event，拆分 context、allocation、H2D、kernel、synchronize、D2H 與 free。
- [ ] 保存測試機 CPU、GPU、RAM、Driver、Toolkit、recipe、影像資訊與 commit hash，建立可重現 baseline。
- [ ] 分開公布冷啟動、warm-up、純檢測與包含輸出的端到端數據。
- [ ] benchmark 記錄平均、P50、P95、CPU/GPU utilization、VRAM、溫度與功耗。

### CPU 與 fallback 正確性

- [x] 缺少 DLL 時，CPU fallback 與純 CPU 的 PASS/NG、tiles、defects、bbox 與 metadata 完整一致。
- [x] fused GPU 呼叫失敗時不採用部分結果，整個 detector 重新從 CPU preprocess 開始執行。
- [x] 建立固定 random seed 合成測例：BGR、gray、全黑、全白、棋盤格與邊界像素。
- [ ] 補入固定真實 AOI 影像測例；待取得可追蹤的生產樣本後執行。
- [x] 覆蓋奇數尺寸、極小圖、4K、non-contiguous stride、1/3 channels 與不同 ROI 尺寸。
- [ ] 五個 production recipes 各準備至少一張 PASS 與一張 NG 樣本。
- [ ] 實機注入 kernel error、CUDA 初始化失敗與 OOM，確認 fallback 後無 stale pointer 或錯誤中間結果。
- [x] `fallback_to_cpu: false` 且 CUDA DLL 不可用時必須明確失敗，不可回報假的 GPU success。

## P1：共用 Preprocess Plan 架構

### Python/OOP execution layer

- [x] 建立 immutable `PreprocessPlan`。
- [x] 建立 typed operators：Gray、Resize、Gaussian、Threshold、AdaptiveMean、Morphology。
- [x] 建立 `CpuPreprocessExecutor`，OpenCV 結果作為 fallback 與等價基準。
- [x] 建立 `CudaPreprocessExecutor`，支援 stateless primitives 與既有 401-2 fused compatibility adapter。
- [x] `BaseDetector.execute_preprocess_plan()` 統一選擇 executor。
- [x] 401-2 改為宣告 Gray → Gaussian → AdaptiveMean，不再直接呼叫 CUDA export。
- [x] CUDA 尚不能保持 `INTER_AREA` 語意時拒絕執行 Resize(area)，不可靜默改用 nearest-neighbor。
- [x] 將 plan 建立移出每次 tile 熱路徑，依 detector params、dtype 與 shape cache immutable plan，並以 bounded LRU 避免無限成長。
- [x] 加入 versioned operator/plan signature、輸入輸出 uint8 型別、channel、shape、順序與 operator 參數 validation。
- [x] 加入 versioned capability report，清楚列出 plan 為何走 fused、primitive、CPU 或 fallback，並寫入 detector execution metadata。

### Generic native plan ABI

- [ ] 定義 versioned C structs：operator kind、input node、參數、output node；不得包含 detector ID/name。
- [ ] 新增 optional `vf_plan_create/execute/destroy` exports；ABI v1 primitives 與 401-2 adapter 保持相容。
- [ ] plan create 階段驗證 operators、channel、shape、參數與輸出；execute 階段只處理資料。
- [ ] 將 Gray、Gaussian、AdaptiveMean、Threshold、Morphology 接入通用 native executor。
- [ ] 通用 native plan 達成一次 H2D、連續 kernels、最後一次必要 D2H。
- [ ] 加入 plan capability query；任一 operator 不支援時整份 plan CPU fallback，避免反覆 CPU/GPU 傳輸。
- [ ] 實作與 OpenCV 等價的 `INTER_AREA` resize 後，才開放 CUDA Resize(area)。
- [x] Python/CPU plan 擴充 topologically ordered DAG/multi-output，支援一份 gray 產生多張 masks。
- [ ] CUDA/native plan 擴充 DAG/multi-output，讓 device gray 直接產生多張 masks。

## P2：CUDA kernels 與資源生命週期

### 已完成的核心 kernels

- [x] Gaussian 使用 horizontal/vertical separable kernels 與 float 中間 buffer。
- [x] Gaussian weights 使用 constant memory。
- [x] Adaptive Mean 使用 replicate-border 64-bit integral image，視窗查詢為 O(1)。
- [x] Integral image 使用 row scan、transpose、第二次 row scan，並檢查 allocation overflow。
- [x] 驗證工具已加入 Gaussian、Adaptive Mean、401-2 fused 與 4K benchmark 案例。
- [ ] Gaussian 加入 shared-memory tile/halo，實測 kernel 45 收益與限制。
- [ ] CUDA event 分別量測 integral 建立、Gaussian passes 與 threshold kernel。

### Persistent context 與 buffers

- [x] 保留 ABI v1 host-pointer primitives，使用 optional export probe 相容舊 DLL。
- [x] 新增 `vf_context_create/destroy/stats`。
- [x] context 擁有 grow-only uint8、float Gaussian 與 64-bit integral buffers。
- [x] 相同或較小尺寸的 401-2 fused 呼叫不再重複 `cudaMalloc/cudaFree`。
- [x] `GpuRuntime` 提供 `close()`、context manager、destructor 與 `RLock` 序列化。
- [ ] 將 CUDA stream、morphology ping-pong 與所有 plan scratch 納入同一 context。
- [ ] monitor/batch 跨多張影像重用同一個長生命週期 `GpuRuntime`/context。
- [ ] 測試尺寸增減、channel 切換、參數改變、CUDA error/OOM 後的重用與釋放。
- [ ] 評估 `cudaMallocAsync`/memory pool；只有相容且實測有收益時採用。

### Morphology

- [ ] 量測 detector 401 多 iterations 的 morphology 占比。
- [ ] 評估矩形 kernel 的 horizontal/vertical separable min/max filter。
- [ ] 多 iterations 使用 device ping-pong buffers，中間不得回傳 CPU。
- [ ] 小 kernel/少 iterations 建立 CPU/GPU crossover 規則。

## P3：Detector 遷移與 CPU/GPU 邊界

- [x] 401-1 遷移到 cached 共用 plan：Gray → Resize(area) → Gaussian → AdaptiveMean → Morphology；CUDA 無法保持 area 語意時整個 detector CPU fallback。
- [x] 401 遷移到 cached 共用 plan，保留 BGR Gaussian → Morphology → Gray → AdaptiveMean、threshold 與 contour 語意。
- [x] 401-2 preprocessing 已遷移到共用 plan，並保留 fused/legacy/CPU 路徑。
- [x] 900 遷移成 cached CPU DAG plan，共用一次 gray 產生 outer global 與 inner adaptive masks。
- [ ] 900 DAG 接上 CUDA/native executor，共用 device gray 並只下載必要 masks。
- [ ] 401/401-1/401-2 的 `findContours` 與少量幾何分析暫留 CPU，只下載 binary mask。
- [x] 401-2 contour mask 改為局部 bbox mask，避免每個 contour 配置整張 ROI mask。
- [ ] 評估 401-2 white-pixel reduction 移至 GPU，只下載統計值與必要 mask。
- [ ] 評估 connected components；只有 bbox/area/排序語意等價且 profiler 證明有收益時取代部分 contours。
- [ ] 全部 detector 遷移並通過 RTX 3090 驗收後，才評估移除 detector-specific compatibility adapter。

## P4：Tiling、ROI、Batch 與跨圖片重用

- [x] 偵測重複 GPU crop round trips，記錄傳輸量並輸出負優化警告。
- [ ] production/benchmark 在 device tiling 改善前預設關閉 GPU crop。
- [ ] 原圖一次 upload，以 device offset/view 表示 grid ROI，不再每 tile 上傳完整原圖。
- [ ] detector 可直接消費 device ROI；只有 CPU contour、GUI、debug 或存檔時才下載。
- [ ] 新增 batch ROI API，以座標陣列產生連續 device buffers。
- [ ] 新增 `run_batch(images/rois)` 或等價 detector batch 介面；CPU 預設實作可逐張執行。
- [ ] 依影像尺寸與可用 VRAM 自動選 batch size；RTX 3090 測試 8、16、32、64 ROI。
- [ ] 單張 GUI 採低延遲策略；資料夾、monitor、batch 採高吞吐策略。
- [ ] 使用 bounded 單一 GPU queue，避免多個 CPU workers 同時搶 GPU 或無限制累積 VRAM。
- [ ] 評估 pinned host memory 與 CUDA streams，量測 upload/kernel/download 重疊收益。

## P5：CPU 與整體 Pipeline 最佳化

- [ ] 分別量測 `findContours`、幾何分析、Python tile/detector 迴圈、progress callback、aggregation 與 reporter。
- [ ] 降低 progress callback 頻率，避免每個小 primitive 更新 GUI。
- [ ] 移除不必要的 `image.copy()`、`np.ascontiguousarray()` 與完整尺寸 temporary masks。
- [ ] 相同 tile 被多個 detectors 使用時，共用 gray 與可重用的 CPU/GPU preprocessing 結果。
- [ ] 對小圖、小 ROI、少 tiles 建立 CPU/GPU crossover benchmark；低於門檻自動選 CPU。
- [ ] Overlay、NG tiles、CSV/JSON 與純檢測計時分離；必要時將檔案輸出移至背景工作。
- [ ] Pattern matching 只有在 profiler 證明為主要熱點後才 GPU 化，模板常駐 GPU 並保留 CPU 等價路徑。
- [ ] PNG 編碼、YAML、彙總、logging 與 GUI 控制邏輯維持 CPU，除非量測證明需要改變。

## P6：GUI、設定與部署

- [x] Recipe 與 GUI 可設定 GPU，並顯示 DLL/device/fallback 狀態。
- [ ] GPU mode 統一為清楚的 `auto`、`cpu`、`cuda` 語意，預設值需由實機驗收決定。
- [ ] GUI worker 不得在 UI thread 等待 CUDA；取消、錯誤與進度更新必須保持可回應。
- [ ] GUI 顯示實際 backend，不得因 recipe 勾選 GPU 就顯示 CUDA active。
- [ ] PyInstaller 包含 `gpu/visionflow_cuda.dll`，但 CPU-only 電腦沒有 DLL/GPU 仍可正常啟動。
- [ ] 有 GPU、無 GPU、DLL 缺少、DLL 版本不符、fallback 開/關各完成一次打包實機測試。

## P7：CI、GitHub Actions 與發布

- [ ] 一般 Windows runner 執行 unit tests、compileall、recipe/CLI/GUI smoke 與 CUDA headers/API 靜態檢查。
- [ ] DLL 與 test EXE 使用明確 source manifest 分開編譯，不以 glob 無差別加入所有 `.cu`。
- [ ] workflow 明確加入 `gpu/include/`，上傳 DLL、LIB、test EXE 與 build log artifacts。
- [ ] CUDA runtime、CPU/GPU 等價、VRAM leak 與 benchmark 只在 GPU self-hosted runner 執行。
- [ ] self-hosted runner 使用 `self-hosted`、`Windows`、`X64`、`gpu`、`rtx3090` labels。
- [ ] 不允許不受信任的 fork PR 直接在可接觸本機資料的 self-hosted runner 執行。
- [ ] GPU job 支援手動與 nightly；PR 至少完成 compile/static checks。
- [ ] 保存 benchmark JSON、Nsight report、Driver/Toolkit/GPU 與 commit hash，支援 commit 間比較。

## RTX 3090 編譯與實機驗收

### 環境與編譯

- [ ] `nvidia-smi` 可看到 GPU，並記錄 Driver、CUDA compatibility 與 VRAM。
- [ ] 安裝 CUDA Toolkit、VS 2022 C++ Build Tools、Windows SDK；確認 `nvcc --version` 與 `where.exe cl`。
- [ ] 使用 x64 Native Tools PowerShell 執行 `gpu/build_cuda_dll.ps1 -Architecture sm_86`。
- [ ] 產生 `visionflow_cuda.dll`、`visionflow_cuda.lib` 與 `test_cuda_api.exe`，沒有 link/architecture 錯誤。
- [ ] `test_cuda_api.exe` 驗證 ABI、device、compute capability、grayscale、context 與 fused smoke。
- [ ] `dumpbin /exports` 檢查所有預期 `vf_` exports；`dumpbin /dependents` 無缺少依賴。

### Primitive、plan 與效能

- [ ] BGR→RGB、crop、threshold、morphology 與 CPU 完全一致。
- [ ] BGR→Gray、resize、Gaussian 與 Adaptive Mean 通過既定像素容差。
- [ ] Gaussian 覆蓋 kernel 3/5/15/25/45 與 structured/non-contiguous inputs。
- [ ] Adaptive Mean 覆蓋 block 3/11/35、正負與小數 C、invert 及邊界輸入。
- [ ] 401-2 fused 與 CPU plan 結果在容差內；相同尺寸連續執行 allocation count 不增加。
- [ ] 通用 native plan 完成後，逐 operator 與完整 plan 對 CPU executor 建立等價矩陣。
- [ ] 記錄 4K primitives、preprocessing plan、純檢測與端到端 CPU/GPU speedup。
- [ ] 連續執行三次完整驗證，沒有 CUDA error、崩潰或 VRAM 持續成長。

### Production recipes、GUI、打包與壓測

- [ ] `PRODUCT_A_AOI_01.yaml` PASS/NG 樣本一致。
- [ ] `PRODUCT_A_CIRCLE_401_1_AOI_01.yaml` PASS/NG 樣本一致。
- [ ] `PRODUCT_A_NEGATIVE_401_AOI_01.yaml` PASS/NG 樣本一致。
- [ ] `PRODUCT_A_WHITE_RATIO_401_2_AOI_01.yaml` PASS/NG 樣本一致。
- [ ] `PRODUCT_A_FRAME_900_AOI_01.yaml` PASS/NG 樣本一致。
- [ ] 比較 tiles、PASS/NG、defect count、bbox、area、confidence、metadata 與 fallback log。
- [ ] GUI 的 recipe 儲存/載入、viewer backend、status、overlay、輸出與 fallback 正確。
- [ ] 打包版在有 NVIDIA GPU 與無 NVIDIA GPU 電腦均完成驗證。
- [ ] warm-up 5 張後測 10、100、1000 張；VRAM 穩定、GUI 可回應、無 crash/error。

## 未來 AI Detector

- [ ] 導入模型時比較 PyTorch CUDA、ONNX Runtime CUDA 與 TensorRT 的部署及效能。
- [ ] 模型/session 只載入一次並常駐 GPU；支援 batch inference 與固定輸入尺寸。
- [ ] 優先驗證 FP16；INT8 必須完成校正與精度驗收後才能啟用。
- [ ] AI 與傳統 CV 共用 GPU scheduler、VRAM budget、warm-up、metrics 與 fallback policy。
- [ ] 避免 GUI、monitor、batch worker 各自載入一份大型模型。

## 最終驗收門檻

- [ ] CPU-only 是完整受支援模式，沒有 CUDA/NVIDIA GPU 仍可啟動 GUI、CLI、batch 與 monitor。
- [ ] 五個 production recipes 通過 CPU/GPU 等價規則，沒有未解釋的 fallback。
- [ ] 每個 GPU plan 原則上每張輸入最多一次 upload 與一次必要 download。
- [ ] warm-up 後不再為每個 operator 重複 `cudaMalloc/cudaFree`。
- [ ] 連續 1000 張後 VRAM 位於穩定平台，沒有資源洩漏或程序崩潰。
- [ ] GPU 純檢測 median 與 P95 在目標資料集均優於 CPU；目標加速門檻為至少 1.5 倍。
- [ ] 未達效能門檻的 recipe/operator 保持 CPU 或 GPU 預設關閉。
- [ ] 加速不得犧牲 GUI 回應、打包啟動、結果追溯、錯誤訊息或 CPU fallback。

## 完成紀錄

- [x] 2026-07-13：建立 `cuda_practice/`、RTX 3090 `sm_86` 練習與編譯說明。
- [x] 2026-07-14：加入 recipe/detector/GUI GPU 開關、CUDA 狀態與安全 CPU fallback。
- [x] 2026-07-14：建立 CUDA DLL C ABI、ctypes bridge、build script、C++ smoke 與 Python 驗證工具。
- [x] 2026-07-14：完成 M0 第一批 profiler、Reporter 分項計時、傳輸統計、crop 警告及容差修正。
- [x] 2026-07-14：完成 CPU-only/缺 DLL fallback 等價回歸。
- [x] 2026-07-14：完成 M1 原始碼：separable Gaussian、constant weights、64-bit integral Adaptive Mean 與 structured tests。
- [x] 2026-07-14：完成 M2 第一個垂直切片：persistent context、grow-only buffers、context stats 與 401-2 fused preprocessing。
- [x] 2026-07-14：建立通用 `PreprocessPlan`、typed operators、CPU/CUDA executors，並遷移 401-2。
- [x] 2026-07-14：將 CPU、GPU、CUDA、GUI、CI、打包與 RTX 3090 驗收清單合併為唯一 `Todo.md`。
- [x] 2026-07-14：更新 `AGENT.md`，統一 Todo 紀律、模組責任、PreprocessPlan/CPU fallback 架構、驗證矩陣、安全 staging 與 commit/push 規範。
- [x] 2026-07-14：更新 `aoi-verify-push`，並新增 `aoi-detector-development`、`aoi-cuda-validate`、`aoi-release` skills；四者皆完成 metadata 與官方 validator 檢查。
- [x] 2026-07-15：整理 2026-07-09 至 2026-07-15 Git 紀錄、GPU/CUDA 進度與待驗收項目，完成本週流水帳報告。
- [x] 2026-07-15：更新並完整中文化根目錄 `README.md`，同步目前 CLI、GUI、配方、Detector、輸出、CUDA fallback、打包與驗證方式。
- [x] 2026-07-17：加入 GUI 預覽影像載入、色彩轉換、QImage/QPixmap、scene 顯示與使用者實際等待時間量測，並輸出至日誌及 viewer backend tooltip。
- [x] 2026-07-17：補齊固定 seed 合成影像、奇數/極小/4K、non-contiguous、1/3 channels、ROI 尺寸 CPU 測試，並驗證關閉 fallback 時缺少 CUDA DLL 會明確失敗；真實照片與 GPU 實機項目保留待辦。
- [x] 2026-07-17：新增 per-detector bounded LRU `PreprocessPlanCache`，依 shape、dtype 與參數 signature 重用 immutable plan；401-2 已移出逐 tile plan 建立熱路徑並加入 cache/失效測試。
- [x] 2026-07-17：加入 versioned operator/plan signature、tensor spec 推導及輸入輸出 dtype/channel/shape/order/參數驗證，CPU 與 CUDA executor 共用相同契約並以 fake runtime 覆蓋錯誤輸出。
- [x] 2026-07-17：加入 preprocess capability report，記錄 requested/selected backend、fused/primitive/CPU/fallback route、原因、plan signature 與不支援項目，並帶入 detector execution metadata。
- [x] 2026-07-17：將 401-1 遷移到 cached shared plan（Gray/Resize area/Gaussian/AdaptiveMean/Morphology），保留 process scale、ROI、contour 與 metadata 語意，area 不支援時維持 full-detector CPU fallback。
- [x] 2026-07-17：將 401 遷移到 cached shared plan，保留 BGR Gaussian/Morphology 後轉 Gray/AdaptiveMean 的既有逐像素順序，以及 ROI、contour、座標、排序與 metadata 語意。
- [x] 2026-07-17：新增 CPU DAG/multi-output plan/executor，900 改以 cached DAG 共用一次 Gray 產生 outer Threshold 與 inner AdaptiveMean masks；CUDA DAG/device gray 另列待辦。
- [x] 2026-07-17：401-2 contour white-ratio 改用局部 bbox mask，避免逐 contour 配置整張 ROI；CPU 測試確認逐像素統計、排序、ROI offset 與 metadata 均與舊 full-ROI 演算法一致。
