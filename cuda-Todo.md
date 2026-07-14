# VisionFlow CUDA DLL 編譯與全流程測試 TODO

這份清單是在另一台有 NVIDIA GPU 的 Windows 電腦上執行。RTX 3090 的架構是 `sm_86`。

## 1. 必要環境

- [ ] 安裝支援目標 CUDA Toolkit 的 NVIDIA Driver，重新開機後 `nvidia-smi` 可看到 GPU。
- [ ] 安裝 CUDA Toolkit（包含 `nvcc`、headers 與 runtime libraries）。
- [ ] 安裝 Visual Studio 2022 Build Tools，勾選「使用 C++ 的桌面開發」及 Windows 10/11 SDK。
- [ ] 使用 **x64 Native Tools PowerShell for VS 2022** 開啟專案。
- [ ] 執行 `nvcc --version`，確認 Toolkit 版本。
- [ ] 執行 `where.exe cl`，確認 MSVC compiler 可找到。
- [ ] 執行 `nvidia-smi`，記錄 Driver、CUDA compatibility、GPU 型號與 VRAM。
- [ ] 專案虛擬環境已安裝 `requirements.txt`；`env\Scripts\python.exe` 可 import `cv2`、`numpy`、`yaml`。

## 2. CUDA DLL 檔案確認

- [x] `gpu/visionflow_cuda.cu`：正式 DLL kernels 與 C ABI 實作。
- [x] `gpu/include/visionflow_cuda.h`：公開 C ABI、參數與 morphology enum。
- [x] `gpu/include/visionflow_cuda_errors.h`：穩定錯誤碼。
- [x] `gpu/include/visionflow_cuda_internal.cuh`：allocation、copy、kernel error helper。
- [x] `gpu/test_cuda_api.cu`：獨立 C++ ABI/device/grayscale smoke。
- [x] `gpu/validate_cuda_dll.py`：OpenCV 數值比對、benchmark 與 AOI CPU/GPU 全流程比對。
- [x] `gpu/build_cuda_dll.ps1`：DLL、import library、smoke executable 與測試入口。

## 3. 編譯

在 repository 根目錄執行：

```powershell
.\gpu\build_cuda_dll.ps1 -Architecture sm_86
```

- [ ] 成功產生 `gpu/visionflow_cuda.dll`。
- [ ] 成功產生 `gpu/visionflow_cuda.lib`。
- [ ] 成功產生 `gpu/test_cuda_api.exe`。
- [ ] build output 沒有 unresolved external、architecture 或 MSVC host compiler 錯誤。
- [ ] 若不是 RTX 3090，依 GPU compute capability 改 `-Architecture`，例如 Ada 使用 `sm_89`。

## 4. C ABI 與 DLL 載入測試

```powershell
.\gpu\test_cuda_api.exe
```

- [ ] ABI 顯示 `1`，與 `VF_CUDA_ABI_VERSION` 一致。
- [ ] CUDA device count 大於 0。
- [ ] GPU 名稱及 compute capability 正確。
- [ ] `C ABI and grayscale smoke passed`。
- [ ] 使用 `dumpbin /exports gpu\visionflow_cuda.dll` 確認所有 `vf_` exports 存在。
- [ ] 使用 `dumpbin /dependents gpu\visionflow_cuda.dll` 確認沒有缺少的第三方 DLL。
- [ ] `cudart` 已靜態連結；NVIDIA Driver 提供的 `nvcuda.dll` 不需要複製到專案。

## 5. Primitive 數值與效能測試

M1 已將 Gaussian 改為 separable kernels、Adaptive Mean 改為 64-bit integral image；執行本節前必須用目前 commit 重新編譯 DLL，不可沿用舊 DLL。

```powershell
.\env\Scripts\python.exe .\gpu\validate_cuda_dll.py `
  --dll .\gpu\visionflow_cuda.dll `
  --benchmark 20
```

- [ ] BGR→RGB 與 CPU 完全相同。
- [ ] BGR→Gray 與 OpenCV 最大差值不超過 1。
- [ ] ROI crop 與 NumPy 完全相同。
- [ ] Gray resize 在設定容差內。
- [ ] Gaussian blur 在設定容差內。
- [ ] Global threshold 與 OpenCV 完全相同。
- [ ] Adaptive mean threshold mismatch ratio 不超過 2%。
- [ ] Open/close/dilate/erode 與 OpenCV 完全相同。
- [ ] 記錄 4K BGR→Gray（包含上傳/下載）的平均毫秒數。
- [ ] 記錄 4K Gray Gaussian k45 與 Adaptive Mean block 35 的 CPU/GPU 平均毫秒數及 speedup。
- [ ] 連續執行測試三次，沒有 CUDA error、VRAM 持續成長或程序崩潰。

## 6. 真實 AOI CPU/GPU 全流程比對

請對每個 production recipe 選至少一張 PASS 與一張 NG 影像：

```powershell
.\env\Scripts\python.exe .\gpu\validate_cuda_dll.py `
  --dll .\gpu\visionflow_cuda.dll `
  --image C:\AOI_TEST\sample.png `
  --recipe .\recipes\PRODUCT_A_AOI_01.yaml `
  --benchmark 20
```

或一次編譯並測試：

```powershell
.\gpu\build_cuda_dll.ps1 -RunTests `
  -Image C:\AOI_TEST\sample.png `
  -Recipe .\recipes\PRODUCT_A_AOI_01.yaml
```

- [ ] `PRODUCT_A_AOI_01.yaml` PASS/NG 樣本一致。
- [ ] `PRODUCT_A_CIRCLE_401_1_AOI_01.yaml` PASS/NG 樣本一致。
- [ ] `PRODUCT_A_NEGATIVE_401_AOI_01.yaml` PASS/NG 樣本一致。
- [ ] `PRODUCT_A_WHITE_RATIO_401_2_AOI_01.yaml` PASS/NG 樣本一致。
- [ ] `PRODUCT_A_FRAME_900_AOI_01.yaml` PASS/NG 樣本一致。
- [ ] Tile 數量、座標與 crop bytes 一致。
- [ ] Detector PASS/NG、defect count、bbox、area、confidence 與 metadata 一致。
- [ ] GPU result 的 `execution.gpu.tiling.active` 為 `true`。
- [ ] 所有已勾選 detector 的 `execution.gpu.detectors.<id>.active` 為 `true`。
- [ ] log 沒有 `CPU fallback`；若出現，先修 DLL，不要把 fallback 當成 GPU 測試通過。

## 7. GUI 操作測試

- [ ] 啟動 `python main.py --gui`。
- [ ] Recipe Designer 顯示 `CUDA DLL 可用`、正確 GPU 名稱。
- [ ] 勾選「切小圖使用 GPU」、「GUI 預覽使用 GPU」與 detector GPU。
- [ ] 儲存 recipe、重新載入，所有 GPU 勾選狀態保持。
- [ ] 載入影像後 viewer 顯示 `顯示: CUDA DLL`。
- [ ] 檢測完成狀態列顯示 `CUDA DLL`。
- [ ] Overlay、縮圖、PASS/NG、CSV、JSON、NG tile 都正確。
- [ ] 將 DLL 暫時改名，確認 `fallback_to_cpu: true` 時 GUI 可運作且清楚顯示 CPU fallback。
- [ ] 將 `fallback_to_cpu: false` 後缺少 DLL 必須明確失敗，不可假裝使用 GPU。

## 8. 打包測試

```powershell
.\build_exe.ps1
```

- [ ] PyInstaller output 包含 `gpu/visionflow_cuda.dll`。
- [ ] 有 NVIDIA GPU 的測試機可啟動並真正啟用 CUDA DLL。
- [ ] 沒有 NVIDIA GPU 的測試機可啟動，GPU 未勾選時完全走 CPU。
- [ ] 沒有 NVIDIA GPU 且 GPU 已勾選時，依 recipe 正確 fallback 或明確失敗。

## 9. 壓力與 benchmark

- [ ] 先 warm-up 5 張，再測 10、100 張影像。
- [ ] 分別記錄 CPU 與 GPU 的平均、P50、P95、最大處理時間。
- [ ] 記錄 GPU utilization、VRAM、溫度與功耗。
- [ ] 監控 batch/monitor 長時間執行是否有 VRAM leak。
- [ ] 確認多 worker 情況不會同時破壞 DLL buffer；目前 Python bridge 以 lock 序列化同一 runtime 呼叫。
- [ ] 將結果與 CUDA Toolkit、Driver、GPU、commit hash 一起保存。

## 10. 驗收完成條件

- [ ] 所有 primitive 通過數值容差。
- [ ] 五個 recipes 的 PASS/NG 樣本 CPU/GPU 結果一致。
- [ ] GUI、CLI、batch、monitor 與打包版均完成至少一次實機測試。
- [ ] 不存在未解釋的 CPU fallback。
- [ ] benchmark 證明目標影像尺寸與批量下 GPU 有實際收益；若傳輸成本使 GPU 較慢，保持該項 recipe GPU 開關關閉。
