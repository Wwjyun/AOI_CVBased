# VisionFlow AOI GPU 優化清單

## 目前實測結論

- [x] `visionflow_cuda.dll` 已成功編譯。
- [x] DLL 已在另一台電腦成功載入，GUI/recipe 顯示 CUDA 已啟用。
- [x] 已確認「CUDA active」不等於整條 AOI pipeline 都在 GPU 執行。
- [x] 初次跨機測試的總耗時沒有比 CPU 路徑短，仍有大量 contour、幾何分析、報表與 GUI 工作在 CPU 執行。
- [ ] 保存測試機的 CPU、GPU、RAM、Driver、CUDA Toolkit、影像尺寸、recipe、commit hash 與原始計時結果，建立可重現 baseline。

> 現階段先不要把 GPU 設為所有 recipe 的預設值。只有通過數值等價與端到端效能門檻的路徑才預設啟用；CPU 版持續作為正確性基準及 fallback。

## P0：量測、止血與正確性基線

### 分階段 profiler

- [x] 在 pipeline 結果與 log 分別記錄 recipe setup、image load、tiling、每個 detector、aggregation 與 reporting total 耗時。
- [x] Reporter 分別記錄 overlay、NG tiles、CSV、matrix CSV 與 JSON 的 host wall-clock 耗時。
- [ ] 加入 GUI 顯示、QImage/QPixmap 轉換與使用者實際等待時間的獨立計時。
- [ ] 在 CUDA DLL 或 bridge 記錄 context 初始化、allocation、H2D、kernel、synchronize、D2H 與 free 耗時。
- [x] ABI v1 bridge 先記錄 DLL load、同步呼叫、lock wait、估算 H2D/D2H bytes 與 round-trip 次數，並清楚標示目前無法拆分 DLL 內部階段。
- [x] 每個 CUDA primitive 記錄呼叫次數、估算 upload/download bytes 與同步 wall time。
- [ ] 將冷啟動與 warm-up 後耗時分開；先 warm-up 5 張，再量 10、100 張。
- [ ] 同時記錄 CPU 使用率、GPU utilization、VRAM、平均值、P50、P95 與最大值。
- [ ] benchmark 關閉 overlay、NG tile、CSV、JSON 等輸出後測純檢測，再另測包含完整輸出的端到端時間。
- [x] 修正 `gpu/validate_cuda_dll.py::compare()`：以 `delta > max_diff` 的像素比例套用 mismatch tolerance，並新增容差回歸測試。

### 立即避免負優化

- [ ] 在 GPU crop 改善前，效能測試與 production recipe 暫時關閉 `gpu.tiling`。
- [x] 加入 GPU crop 呼叫與估算 H2D bytes 計數；同一張圖發生多次同步 crop round-trip 時輸出負優化警告。
- [ ] 防止每切一張 tile 都重新上傳完整原圖；改為可重用 source device buffer 後才能移除警告。
- [ ] 對小圖、小 ROI 與少量 tile 建立 CPU/GPU crossover benchmark，低於門檻時自動使用 CPU。
- [x] 保留既有 pipeline `duration_sec` 語義，另外提供 `execution.performance.end_to_end_sec` 與 reporting 明細。

### CPU/GPU 等價測試

- [ ] 建立固定 random seed 的灰階、BGR、全黑、全白、棋盤格、單像素邊界及真實 AOI 測試影像。
- [ ] 覆蓋奇數尺寸、非連續 stride、1/3 channel、極小圖、4K 圖及不同 ROI 尺寸。
- [x] GPU 驗證工具加入 Gaussian kernel 3、5、15、25、45，以及 random/black/white/checker/non-contiguous/BGR 測例；待 RTX 3090 執行確認。
- [x] GPU 驗證工具加入 Adaptive threshold block 3、11、35、正負/小數 C、invert、邊界與 non-contiguous 測例；待 RTX 3090 執行確認。
- [ ] Morphology 覆蓋 open、close、dilate、erode、不同 kernel 與多 iterations。
- [ ] 五個 production recipes 各準備 PASS/NG 樣本，檢查 tile、PASS/NG、defect count、bbox、area、confidence 與 metadata。
- [x] 新增缺少 DLL 時 CPU fallback 與純 CPU 完整結果一致的自動回歸測試。
- [ ] 使用實機 DLL 注入 kernel error/OOM，驗證任一 GPU 步驟失敗時整個 detector 正確回退 CPU，不可混用半套中間結果。

## P1：先降低 kernel 複雜度

### Adaptive mean threshold 改 integral image

- [x] 將目前每 pixel 掃描 `block_size²` 的 adaptive kernel 改成 integral image，讓每個視窗查詢為 O(1)。
- [x] 實作不依賴第三方 library 的 block row scan、transpose 與第二次 row scan，建立二維 integral image。
- [x] 使用 replicate padded border，並以 CPU 模擬確認 block 3/11/35、正負/小數 C、binary/invert 與 OpenCV 結果完全一致。
- [x] Integral 累加使用 64-bit unsigned 型別，並檢查 padded dimensions 與 allocation size overflow。
- [ ] 重複使用 integral scratch buffer，不可每次呼叫重新配置。
- [ ] 用 CUDA event 分別記錄 integral 建立與 threshold kernel 耗時。

### Gaussian 改 separable kernel

- [x] 將目前 `kernel_size²` 的二維卷積拆成 horizontal 與 vertical 兩個一維 kernel。
- [x] 使用 float 中間 buffer，避免 horizontal pass 提前量化成 `uint8` 擴大 CPU/GPU 差異。
- [ ] 使用 shared memory tile 與 halo，減少重複 global memory load。
- [x] Gaussian weights 改成 constant memory，移除每次呼叫的 device weights malloc/free。
- [ ] 驗證 kernel 45 等大型 filter 的效能與 shared memory 上限。

### Morphology

- [ ] 量測 morphology 在 detector 401 多 iterations 下的占比。
- [ ] 對矩形 structuring element 評估 horizontal/vertical separable min/max filter。
- [ ] 重複 iterations 時維持 ping-pong device buffers，不可在每個 iteration 回傳 CPU。
- [ ] 小 kernel 與少 iterations 若 CPU 較快，保留自動 CPU crossover。

## P1：Persistent GPU context 與單次傳輸 pipeline

### ABI 與資源生命週期

- [x] 保留現有 ABI v1 host-pointer API；新功能採 optional export probe，舊 DLL 仍可走 stateless primitives。
- [x] 新增 opaque `vf_context_create/destroy/stats` API 與 DLL 內部 grow-only typed buffer reserve。
- [x] 第一版 context 擁有 reusable uint8、float Gaussian、64-bit integral scratch buffers 與 allocation/reserved-bytes counters。
- [ ] 將 CUDA stream 與其餘 detector 所需 ping-pong buffers 納入同一 context。
- [x] buffer 容量不足時才成長；相同或較小尺寸的後續 fused 呼叫不再 `cudaMalloc/cudaFree`。
- [x] `GpuRuntime` 提供明確 `close()`、context manager 與 destructor 安全釋放。
- [ ] 讓 monitor/batch 的 `GpuRuntime` 跨多張圖片重複使用同一 context；目前先在單次 pipeline 的多 tile 間重用。
- [x] 保持目前 `RLock` 序列化 context/fused calls；context 不暴露 raw device pointer 給 Python。
- [ ] 測試尺寸變大、變小、切換 channel、CUDA error、OOM 與 fallback 後沒有 stale pointer 或 VRAM leak。

### Detector fused preprocessing

- [ ] 401：單次 upload 後在 GPU 完成 Gaussian → morphology → gray → adaptive threshold，只下載 binary mask。
- [ ] 401-1：單次 upload 後完成 gray → resize → Gaussian → adaptive threshold → morphology，只下載 binary mask。
- [x] 401-2：單次 upload BGR/gray 後完成 gray → Gaussian → integral adaptive threshold，只下載 binary mask。
- [ ] 900：gray 只建立一次，在同一份 device gray 上產生 outer global mask 與 inner adaptive mask，再一次下載兩張 mask。
- [x] 新增 fused failure 回歸：失敗時不採用部分 mask，`BaseDetector` 將整個 detector 重跑 CPU。
- [ ] 在 RTX 3090 注入實際 CUDA kernel/OOM error，確認 error state、context 與後續 CPU/GPU 呼叫可恢復。
- [x] 驗證工具加入 fused 401-2 CPU 等價、相同尺寸 allocation count 不成長，以及 4K CPU/GPU speedup benchmark。

## P1：Tiling 與 ROI 資料流

- [ ] 不再讓 `vf_crop_u8` 每個 tile 上傳一次完整原圖。
- [ ] grid 模式先上傳完整原圖一次，再用 device offset/view 表示多個 ROI。
- [ ] detector 可直接消費 device ROI；若下一步仍在 GPU，不要先把 crop 下載 CPU 再重新上傳。
- [ ] 加入 batch ROI crop API，一次提交座標陣列並輸出到連續 device buffer。
- [ ] pattern-match 與 anchored-grid 在 CPU 找到座標後，將座標批次交給 GPU，不逐 ROI 同步。
- [ ] 只有 GUI、存檔、CPU contour 或 debug 真正需要像素時才下載 tile/mask。
- [ ] 若 detector 仍完全走 CPU，該 tile 維持 NumPy crop，不強迫經過 GPU。

## P2：Batch、非同步傳輸與跨圖片重用

- [ ] 新增 `run_batch(images/rois)` 或等價的 detector batch API。
- [ ] 依影像尺寸和可用 VRAM 自動選 batch size；RTX 3090 先測 8、16、32、64 個 ROI。
- [ ] 使用 pinned host memory 測試 H2D/D2H；確認收益後才保留，並限制 pinned memory 總量。
- [ ] 使用 CUDA streams 重疊下一批 upload、目前 batch kernel 與上一批 download。
- [ ] 使用 CUDA events 計時，不用 CPU wall clock 推估非同步 kernel 時間。
- [ ] monitor/batch 模式共用長生命週期 `GpuRuntime`，避免每張圖重新載入 DLL、初始化 CUDA 與配置 buffer。
- [ ] 多個 CPU worker 透過單一 bounded GPU queue 提交工作，避免同時搶 GPU 或無限制累積 VRAM。
- [ ] 評估 `cudaMallocAsync`/memory pool；只有在支援的 Driver/Toolkit 與實測有收益時啟用。

## P2：降低 CPU/GPU 邊界成本

- [ ] 401-2 評估在 GPU 完成 white-pixel count/reduction，只回傳統計值與必要 mask。
- [ ] 評估 connected components 是否能取代部分 `findContours` 使用案例；先確認 bbox/area/排序語義與 CPU 版等價。
- [ ] 對仍需 `findContours` 的 detector，只下載最小 binary mask，不下載不再使用的 BGR/gray 中間圖。
- [ ] 900 評估在 GPU 做候選 bbox/area reduction；若候選很少或 CPU 已非瓶頸則保持 CPU。
- [ ] Pattern matching 只有在 profiler 證明為主要熱點後才移入 DLL，並保留相同 score/NMS 語義。

## P3：CPU 與整體 pipeline 優化

- [ ] 分別量測 `findContours`、contour 幾何分析、Python tile/detector 迴圈、progress callback、aggregation 與 reporter。
- [ ] 減少高頻 progress callback；以 tile 或固定時間間隔更新 GUI，不對每個小 primitive 更新。
- [ ] 避免不必要的 `image.copy()`、`np.ascontiguousarray()` 與完整尺寸 temporary mask。
- [ ] 401-2 contour mask 改成局部 bbox mask，避免每個 contour 都配置整張 ROI 大小的 mask。
- [ ] 輸出 overlay、NG tile、CSV/JSON 與純檢測計時分離；需要時讓檔案輸出在背景執行。
- [ ] 只有 profiler 證明 CPU contour/geometry 是主要瓶頸後，才評估 GPU connected components 或自訂幾何 kernel。

## GitHub Actions 與實機驗證

- [ ] 一般 Windows runner 執行 Python compileall、recipe/GUI smoke、CUDA headers/API 靜態檢查與可行的 CUDA compile job。
- [ ] DLL 與 test EXE 使用明確 source manifest 分開編譯，不用 glob 將所有 `.cu` 無差別編入 DLL。
- [ ] workflow 明確加入 `gpu/include/`，並上傳 DLL、LIB、test EXE、build log 作為 artifacts。
- [ ] 真正 CUDA runtime、CPU/GPU 等價、VRAM leak 與 benchmark 放在帶 GPU 的 self-hosted runner。
- [ ] self-hosted runner 使用 `self-hosted`、`Windows`、`X64`、`gpu`、`rtx3090` 等 labels，避免工作被送到沒有 GPU 的 runner。
- [ ] GPU job 支援手動觸發與 nightly；PR 至少跑 compile，合併前或 nightly 跑完整 GPU regression。
- [ ] 不允許不受信任的 fork PR 在可接觸本機資料的 self-hosted runner 上直接執行。
- [ ] 保存 benchmark JSON、Nsight Systems report、GPU/Driver/Toolkit 與 commit hash，讓不同 commit 可比較。

## 驗收門檻

- [ ] 五個 production recipes 的 PASS/NG、tile、defect count、bbox 與 metadata 通過既定 CPU/GPU 等價規則。
- [ ] warm-up 後不再出現每個 primitive 的 `cudaMalloc/cudaFree`，每個 detector 每張 tile 原則上最多一次 upload 與一次必要 download。
- [ ] 連續 1000 張後 VRAM 回到穩定平台值，沒有持續成長、CUDA error 或程序崩潰。
- [ ] 分別公布 kernel-only、detector preprocessing、純檢測 pipeline、包含輸出的端到端數據，不用單一數字混在一起。
- [ ] GPU 路徑在目標測試集的純檢測 median 與 P95 都優於 CPU；未達標的 recipe 保持 GPU 預設關閉。
- [ ] 加速不能犧牲 CPU fallback、GUI 回應、打包版啟動或結果可追溯性。

## 建議實作順序

- [ ] M0：加入 profiler、修正測試判斷、建立跨機 CPU/GPU baseline。
- [ ] M1：停用負優化的 GPU tiling，完成 integral adaptive threshold 與 separable Gaussian。
- [ ] M2：建立 persistent context/buffers 與 401/401-1/401-2/900 fused preprocessing。
- [ ] M3：改為原圖單次 upload、device ROI、batch ROI 與跨圖片 runtime 重用。
- [ ] M4：加入 pinned memory、streams、VRAM-aware batch，再依 profiler 處理剩餘 CPU 熱點。
- [ ] M5：完成 RTX 3090 regression、長時間壓測、GitHub Actions GPU job 與 production 預設值評估。

## Progress / Completed Items

- [x] 2026-07-14 完成 M0 第一批觀測能力：OOP `PipelineProfiler`、ABI v1 host-side CUDA metrics、GPU crop 負優化警告、Reporter 分項計時及容差判斷修正。
- [x] 2026-07-14 新增 CPU-only 與缺少 GPU/DLL fallback 等價回歸；確認 PASS/NG、tiles、defects 與 metadata 不因觀測層改變。
- [x] 2026-07-14 完成 M1 CUDA 原始碼：separable Gaussian、constant weights、64-bit integral adaptive threshold、replicate border 與擴充驗證矩陣；公開 C ABI 與 CPU fallback 保持不變。
- [ ] 在 RTX 3090 重新編譯 M1 DLL，執行新增 primitive matrix、4K CPU/GPU benchmark 與五個 recipe 全流程等價測試後，才能將 M1 標示完成。
- [x] 2026-07-14 完成 M2 第一個垂直切片：optional persistent context ABI、grow-only buffers/context stats、401-2 fused BGR/gray preprocessing、舊 DLL legacy GPU 路由與 fused failure 全 detector CPU fallback。
