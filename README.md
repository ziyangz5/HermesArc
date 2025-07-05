# HermesArc Library  
[![MIT license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ziyangz5/HermesArc/blob/main/LICENSE)

HermesArc is an interpolation library that supporting data transfer between Unity DX12 RenderTexture and [LibTorch](https://pytorch.org/docs/stable/cpp_index.html). With HermesArc, you can efficiently move frames directly to CUDA/Torch on the GPU, avoiding unnecessary CPU tour.

## Install

1. Download the most recent Unity plugin from [Releases](https://github.com/ziyangz5/HermesArc/releases).
2. Put the HermesArc.dll file into Assets/Plugin/
3. Enjoy

## How to use

We will provide sample code later. Currently, you can take this as a reference:

```C#
using System;
using System.Runtime.InteropServices;
using System.Threading;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UI;                             
using UnityEngine.Experimental.Rendering;         
using AOT;
using System.Collections;

public class UnityDx12CudaAsync : MonoBehaviour
{
    private const string PluginName = "HermesArc";

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void FrameReadyCallbackType(float mean);

    [DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
    static extern void RegisterFrameReadyCallback(FrameReadyCallbackType cb);

    [DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
    static extern void CaptureFrameAsync(
        IntPtr nb0, IntPtr nb1, IntPtr nb2, IntPtr nb3,
        uint width, uint height);

    [DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
    static extern void PushLastTensorToUnity(
        IntPtr nb, uint width, uint height);

    [DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
    static extern bool InitializeNeuralNetwork(string modelPath);

    [DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
    static extern IntPtr get_version();

    [Header("Assign in Inspector")]
    public Camera sourceCamera;    // your camera for CaptureFrameAsync
    public RawImage outputUI;      // the RawImage you want to drive

    RenderTexture floatRT;
    Material gammaMaterial;
    FrameReadyCallbackType frameCb;
    static SynchronizationContext unityContext;

    void Start()
    {
        unityContext = SynchronizationContext.Current;

        string ver = Marshal.PtrToStringAnsi(get_version());
        Debug.Log($"[DX12CudaAsync] HermesArc v{ver}");
        if (!InitializeNeuralNetwork("simple_rotation_512x512.pth"))
            Debug.LogError("[DX12CudaAsync] NN load failed");

        int w = Screen.width, h = Screen.height;
        var desc = new RenderTextureDescriptor(w, h,
            GraphicsFormat.R32G32B32A32_SFloat,
            depthBufferBits: 0);
        desc.enableRandomWrite = true;
        desc.sRGB = false;  // linear
        floatRT = new RenderTexture(desc);
        floatRT.Create();

        var shader = Shader.Find("UI/GammaCorrected");
        if (shader == null)
            Debug.LogError("Could not find UI/GammaCorrected.shader");
        gammaMaterial = new Material(shader);

        if (outputUI != null)
        {
            outputUI.texture = floatRT;
            outputUI.material = gammaMaterial;
        }

        frameCb = OnFrameReady;
        RegisterFrameReadyCallback(frameCb);
        StartCoroutine(CaptureLoop(0.014286f));
    }

    IEnumerator CaptureLoop(float dt)
    {
        while (true)
        {
            yield return new WaitForSeconds(dt);

            var rtN = Shader.GetGlobalTexture("_MyGBufferNormalCapture") as RenderTexture;
            var rtD = Shader.GetGlobalTexture("_MyGBufferDepthCapture") as RenderTexture;
            var rtT = Shader.GetGlobalTexture("_MyGBufferTexCapture") as RenderTexture;
            var rtS = sourceCamera.targetTexture;

            CaptureFrameAsync(
                rtN.GetNativeTexturePtr(),
                rtD.GetNativeTexturePtr(),
                rtT.GetNativeTexturePtr(),
                rtS.GetNativeTexturePtr(),
                (uint)rtN.width, (uint)rtN.height);

            yield return new WaitForSeconds(dt);

            PushLastTensorToUnity(
                floatRT.GetNativeTexturePtr(),
                (uint)floatRT.width, (uint)floatRT.height);
        }
    }

    void OnDestroy()
    {
        RegisterFrameReadyCallback(null);
    }

    [MonoPInvokeCallback(typeof(FrameReadyCallbackType))]
    static void OnFrameReady(float mean)
    {
        if (unityContext != null)
            unityContext.Post(_ => Debug.Log($"Frame mean: {mean}"), null);
        else
            Debug.Log($"Frame mean: {mean}");
    }
}

```
