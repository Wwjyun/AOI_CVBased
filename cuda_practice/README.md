# AOI CUDA 練習範例

這個資料夾依 `Todo.md` 的 GPU 優先順序整理，每支 `.cu` 都可獨立編譯，不依賴 OpenCV。

## 建議順序

1. `00_device_info.cu`：確認 CUDA 與 RTX 3090。
2. `01_grayscale.cu`：BGR 轉灰階。
3. `02_threshold.cu`：固定二值化。
4. `03_gaussian_blur.cu`：3x3 Gaussian Blur。
5. `04_morphology.cu`：3x3 dilation / erosion。
6. `05_template_matching.cu`：SSD 模板比對。
7. `06_roi_batch.cu`：多張同尺寸 ROI 批次二值化。

## Windows 編譯環境

先安裝：

- NVIDIA Driver
- CUDA Toolkit（包含 `nvcc`）
- Visual Studio 2022 Build Tools，勾選「使用 C++ 的桌面開發」

開啟 **x64 Native Tools Command Prompt for VS 2022**，確認：

```bat
nvcc --version
nvidia-smi
```

單獨編譯：

```bat
cd C:\Users\王\Desktop\AOI_CVbased\cuda_practice
nvcc -std=c++17 -O2 -arch=sm_86 00_device_info.cu -o 00_device_info.exe
00_device_info.exe
```

RTX 3090（Ampere）的 compute capability 是 8.6，因此範例使用 `-arch=sm_86`。其他檔案只要替換檔名即可。

批次編譯：

```powershell
.\build_all.ps1
```

若 PowerShell 禁止執行腳本，可在目前視窗暫時允許：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## 練習重點

- `blockIdx`、`blockDim`、`threadIdx` 如何對應像素或 ROI。
- `cudaMalloc`、`cudaMemcpy`、kernel launch、`cudaDeviceSynchronize`、`cudaFree` 的順序。
- 用 `CUDA_CHECK` 檢查錯誤，不要忽略 kernel launch 失敗。
- 先驗證結果正確，再用 CUDA Event 計時。
- 真正整合 AOI 時，應讓影像留在 GPU 串接多個步驟；不要每個 kernel 都重新上傳與下載。

目前專案電腦的 PATH 找不到 `nvcc`，所以這些檔案尚未在本機實際編譯。安裝 CUDA Toolkit 後即可照上述方式練習。
