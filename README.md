# HermesArc Library  
[![MIT license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ziyangz5/HermesArc/blob/main/LICENSE)

HermesArc is an interpolation library that supporting data transfer between Unity DX12 RenderTexture and [LibTorch](https://pytorch.org/docs/stable/cpp_index.html). With HermesArc, you can efficiently move frames directly to CUDA/Torch on the GPU, avoiding unnecessary CPU tour.

## Install

1. Download the most recent Unity plugin from [Releases](https://github.com/ziyangz5/HermesArc/releases).
2. Put the HermesArc.dll file into Assets/Plugin/
3. Enjoy

## How to use

We will provide sample code later. Current, you can take this as a reference:

```C#
using System;
using System.Runtime.InteropServices;
using System.Threading;
using UnityEngine;
using UnityEngine.Rendering;
using AOT;

public class UnityDx12CudaAsync : MonoBehaviour
{
    private const string PluginName = "HermesArc";

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void FrameReadyCallbackType(float mean);

    [DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void RegisterFrameReadyCallback(FrameReadyCallbackType callback);

    [DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void CaptureFrameAsync(IntPtr backbufferPtr, uint width, uint height);

    [DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr get_version();


    [DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void PushLastTensorToUnity(IntPtr backbufferPtr, uint width, uint height);



    public RenderTexture sourceTexture;
    public RenderTexture targetTexture;
    private FrameReadyCallbackType frameReadyDelegate;
    private static SynchronizationContext unityContext;

    bool test = true;
    void Start()
    {

        unityContext = SynchronizationContext.Current;


        var ver = Marshal.PtrToStringAnsi(get_version());
        Debug.Log($"Loaded HermesArc: {ver}");

        if (sourceTexture == null)
        {
            Debug.LogWarning("No sourceTexture assigned; creating 512×512 dummy.");
            sourceTexture = new RenderTexture(512, 512, 0, RenderTextureFormat.ARGB32);
        }


        targetTexture.enableRandomWrite = true;
        targetTexture.Create();
        sourceTexture.enableRandomWrite = true;
        sourceTexture.Create();


        frameReadyDelegate = OnFrameReady;
        RegisterFrameReadyCallback(frameReadyDelegate);
    }

    void FixedUpdate()
    {
        if (targetTexture == null) return;
        GL.IssuePluginEvent(0);
        if (test)
        {
            CaptureFrameAsync(
                sourceTexture.GetNativeTexturePtr(),
                (uint)sourceTexture.width,
                (uint)sourceTexture.height
            );
            test = false;
        }
        else
        {
            PushLastTensorToUnity(
                targetTexture.GetNativeTexturePtr(),
                (uint)targetTexture.width,
                (uint)targetTexture.height
            );
            test = true;
        }

    }

    void OnDestroy()
    {
        RegisterFrameReadyCallback(null);
    }

    [MonoPInvokeCallback(typeof(FrameReadyCallbackType))]
    private static void OnFrameReady(float mean)
    {
        if (unityContext != null)
        {
            unityContext.Post(_ =>
            {
                Debug.Log($"{mean}");

            }, null);
        }
    }
}
```