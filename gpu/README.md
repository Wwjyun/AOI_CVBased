# VisionFlow CUDA DLL

`visionflow_cuda.dll` 是 AOI 的可選 CUDA backend。Recipe 未勾選 GPU 時不會載入或呼叫它；勾選但 DLL/裝置不可用時，預設回退 CPU。

RTX 3090 主機安裝 CUDA Toolkit 後執行：

```powershell
.\gpu\build_cuda_dll.ps1
```

build script 以 `sm_86` 編譯。DLL 需放在 `gpu/visionflow_cuda.dll`，也可由 recipe 的 `gpu.dll_path` 指定其他位置。
