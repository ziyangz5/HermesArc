#include <fstream>
#include <string>
#include <vector>
#include <cuda_runtime_api.h>
#include <torch/torch.h>
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <IUnityInterface.h>
#include <IUnityGraphics.h>
#include <IUnityGraphicsD3D12.h>
#include "library.h"

enum LogLevel {
    LOG_NONE = 0,
    LOG_ERROR = 1,
    LOG_VERBOSE = 2
};

static int g_logVerbosity = LOG_VERBOSE;

static void FileLog(const std::string& msg, int level = LOG_VERBOSE)
{
    if (g_logVerbosity < level)
        return;
    static std::ofstream log("DX12CudaPlugin.log", std::ios::app);
    log << msg << std::endl;
}

extern "C" HERMESARC_API void SetLogVerbosity(int level)
{
    g_logVerbosity = level;
}

typedef void (*FrameReadyCallback)(float);

static IUnityInterfaces*      s_UnityInterfaces = nullptr;
static IUnityGraphics*        s_Graphics        = nullptr;
static IUnityGraphicsD3D12v2* s_GfxD3D12        = nullptr;
static ID3D12Device*          s_Device          = nullptr;
static ID3D12CommandAllocator* s_Allocator      = nullptr;
static ID3D12Resource*        s_Backbuffer      = nullptr;
static ID3D12Resource*        s_StagingTex      = nullptr;
static ID3D12Fence*           s_Fence           = nullptr;
static HANDLE                 s_FenceEvent      = nullptr;
static HANDLE                 g_shareH          = nullptr;
static cudaExternalMemory_t   g_extMem          = nullptr;
static uint64_t               g_rowPitch        = 0;
static size_t                 g_totalBytes      = 0;
static FrameReadyCallback     g_FrameReadyCb    = nullptr;
static HANDLE                 g_WaitHandle      = nullptr;
static unsigned               g_lastWidth       = 0;
static unsigned               g_lastHeight      = 0;
static torch::Tensor          g_lastTensor;
static void noopCudaDeleter(void*) {}

static void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType);
extern "C" HERMESARC_API void CALLBACK FrameFenceCallback(PVOID context, BOOLEAN timedOut);

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces* unityInterfaces)
{
    s_UnityInterfaces = unityInterfaces;
    s_Graphics = s_UnityInterfaces->Get<IUnityGraphics>();
    s_Graphics->RegisterDeviceEventCallback(OnGraphicsDeviceEvent);
    OnGraphicsDeviceEvent(kUnityGfxDeviceEventInitialize);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginUnload()
{
    if (s_Graphics)
        s_Graphics->UnregisterDeviceEventCallback(OnGraphicsDeviceEvent);
    if (s_Allocator)    { s_Allocator->Release();    s_Allocator = nullptr; }
    if (s_StagingTex)   { s_StagingTex->Release();   s_StagingTex = nullptr; }
    if (s_Fence)        { s_Fence = nullptr; }
    if (s_FenceEvent)   { CloseHandle(s_FenceEvent); s_FenceEvent = nullptr; }
    if (g_WaitHandle)   { UnregisterWaitEx(g_WaitHandle, INVALID_HANDLE_VALUE); g_WaitHandle = nullptr; }
    if (g_extMem)       { cudaDestroyExternalMemory(g_extMem); }
    if (g_shareH)       { CloseHandle(g_shareH); }
}

static void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType)
{
    if (eventType == kUnityGfxDeviceEventInitialize)
    {
        if (s_Graphics->GetRenderer() == kUnityGfxRendererD3D12)
        {
            s_GfxD3D12 = s_UnityInterfaces->Get<IUnityGraphicsD3D12v2>();
            if (s_GfxD3D12)
            {
                s_Device = s_GfxD3D12->GetDevice();
                s_Fence = s_GfxD3D12->GetFrameFence();
                if (!s_FenceEvent)
                    s_FenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
                FileLog("[DX12] Acquired D3D12 v2 interface and Unity fence", LOG_VERBOSE);
            }
            else
            {
                FileLog("[DX12] ERROR: D3D12 v2 interface unavailable", LOG_ERROR);
                s_Device = nullptr;
            }
        }
        else
        {
            FileLog("[Graphics] Active API: " + std::to_string((int)s_Graphics->GetRenderer()), LOG_VERBOSE);
            s_Device = nullptr;
        }
    }
    else if (eventType == kUnityGfxDeviceEventShutdown)
    {
        FileLog("[DX12] Graphics device shutdown", LOG_VERBOSE);
        s_Device = nullptr;
        s_GfxD3D12 = nullptr;
    }
}

extern "C" HERMESARC_API void RegisterFrameReadyCallback(FrameReadyCallback cb)
{
    g_FrameReadyCb = cb;
}

extern "C" HERMESARC_API void CaptureFrameAsync(void* backbufferPtr, unsigned width, unsigned height)
{
    s_Backbuffer = reinterpret_cast<ID3D12Resource*>(backbufferPtr);

    if (!s_Allocator && s_Device)
    {
        HRESULT hrAlloc = s_Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&s_Allocator));
        if (FAILED(hrAlloc)) { FileLog("[DX12] CreateCommandAllocator failed: " + std::to_string(hrAlloc), LOG_ERROR); return; }
    }

    if (!s_StagingTex && s_Device)
    {
        D3D12_RESOURCE_DESC desc = s_Backbuffer->GetDesc();
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        hp.CreationNodeMask = 1;
        hp.VisibleNodeMask = 1;

        HRESULT hr = s_Device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&s_StagingTex));
        if (FAILED(hr)) { FileLog("[DX12] CreateCommittedResource failed: " + std::to_string(hr), LOG_ERROR); return; }

        hr = s_Device->CreateSharedHandle(s_StagingTex, nullptr, GENERIC_ALL, nullptr, &g_shareH);
        if (FAILED(hr)) { FileLog("[DX12] CreateSharedHandle failed: " + std::to_string(hr), LOG_ERROR); return; }

        auto allocInfo = s_Device->GetResourceAllocationInfo(0, 1, &desc);
        g_totalBytes = allocInfo.SizeInBytes;
        g_rowPitch = (uint64_t)width * 4 * sizeof(float);

        D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout = {};
        UINT numRows = 0;
        UINT64 rowBytes = 0;
        UINT64 total = 0;
        s_Device->GetCopyableFootprints(&desc, 0, 1, 0, &layout, &numRows, &rowBytes, &total);

        g_rowPitch = layout.Footprint.RowPitch;
        g_totalBytes = total;

        FileLog("D3D12 layout: RowPitch=" + std::to_string(g_rowPitch) + " bytes=" + std::to_string(g_totalBytes), LOG_VERBOSE);

        cudaExternalMemoryHandleDesc memDesc = {};
        memDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
        memDesc.handle.win32.handle = g_shareH;
        memDesc.size = g_totalBytes;
        memDesc.flags = cudaExternalMemoryDedicated;
        cudaError_t impErr = cudaImportExternalMemory(&g_extMem, &memDesc);
        FileLog(std::string("cudaImportExternalMemory -> ") + cudaGetErrorString(impErr) + ", g_extMem=" + std::to_string((uintptr_t)g_extMem), LOG_VERBOSE);
        if (impErr != cudaSuccess) return;
    }

    s_Allocator->Reset();

    ID3D12GraphicsCommandList* cmdList = nullptr;
    HRESULT hrCL = s_Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, s_Allocator, nullptr, IID_PPV_ARGS(&cmdList));
    if (FAILED(hrCL) || !cmdList) { FileLog("[DX12] CreateCommandList failed: " + std::to_string(hrCL)); return; }

    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = s_Backbuffer;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmdList->ResourceBarrier(1, &barrier);

    cmdList->CopyResource(s_StagingTex, s_Backbuffer);

    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    cmdList->ResourceBarrier(1, &barrier);
    cmdList->Close();

    FileLog("[DX12] Command list recorded & closed.", LOG_VERBOSE);

    UINT64 unityFenceValue = s_GfxD3D12->ExecuteCommandList(cmdList, 0, nullptr);
    cmdList->Release();

    g_lastWidth = width;
    g_lastHeight = height;

    s_Fence->SetEventOnCompletion(unityFenceValue, s_FenceEvent);

    RegisterWaitForSingleObject(&g_WaitHandle, s_FenceEvent, FrameFenceCallback, nullptr, INFINITE, WT_EXECUTEONLYONCE);
    FileLog("[DX12] CaptureFrameAsync scheduled callback.", LOG_VERBOSE);
}

extern "C" HERMESARC_API void CALLBACK FrameFenceCallback(PVOID, BOOLEAN)
{
    FileLog(">>> FrameFenceCallback start", LOG_VERBOSE);

    void* devPtr = nullptr;
    cudaExternalMemoryBufferDesc bufDesc = {0, g_totalBytes, 0};

    cudaError_t mapErr = cudaExternalMemoryGetMappedBuffer(&devPtr, g_extMem, &bufDesc);
    FileLog(std::string("cudaExternalMemoryGetMappedBuffer -> ") + cudaGetErrorString(mapErr), LOG_ERROR);
    if (mapErr != cudaSuccess) return;

    std::vector<int64_t> dims    = { (int64_t)g_lastHeight, (int64_t)g_lastWidth, 4 };
    std::vector<int64_t> strides = { (int64_t)(g_rowPitch / sizeof(float)), 4, 1 };

    FileLog("  dims    = [" + std::to_string(dims[0]) + "," + std::to_string(dims[1]) + "," + std::to_string(dims[2]) + "]", LOG_VERBOSE);
    FileLog("  strides = [" + std::to_string(strides[0]) + "," + std::to_string(strides[1]) + "," + std::to_string(strides[2]) + "]", LOG_VERBOSE);

    cudaError_t last = cudaGetLastError();
    FileLog(std::string("  cudaGetLastError before from_blob: ") + cudaGetErrorString(last), LOG_ERROR);

    try {
        auto t1 = torch::from_blob(
                devPtr, dims, strides, &noopCudaDeleter,
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0)
        );
        FileLog("  from_blob succeeded", LOG_VERBOSE);

        last = cudaGetLastError();
        FileLog(std::string("  cudaGetLastError after from_blob: ") + cudaGetErrorString(last), LOG_VERBOSE);

        auto t2 = t1.clone();
        g_lastTensor = t2;
        FileLog("  clone() succeeded", LOG_VERBOSE);
        FileLog(std::to_string(t2[0][0][0].item<float>()), LOG_VERBOSE);
        last = cudaGetLastError();
        FileLog(std::string("  cudaGetLastError after clone: ") + cudaGetErrorString(last), LOG_VERBOSE);

        float meanVal = t2.mean().item<float>();
        FileLog("  mean() = " + std::to_string(meanVal), LOG_VERBOSE);

        if (g_FrameReadyCb) {
            g_FrameReadyCb(meanVal);
            FileLog("  managed callback invoked", LOG_VERBOSE);
        } else {
            FileLog("  no managed callback registered", LOG_VERBOSE);
        }
    } catch (const std::exception& e) {
        FileLog(std::string("  EXCEPTION: ") + e.what(), LOG_ERROR);
    }

    last = cudaGetLastError();
    FileLog(std::string("<<< FrameFenceCallback end, final cudaGetLastError: ") + cudaGetErrorString(last), LOG_ERROR);
}

extern "C" HERMESARC_API void PushLastTensorToUnity(void* renderTexPtr, unsigned width, unsigned height)
{
    ID3D12Resource* dstRes = reinterpret_cast<ID3D12Resource*>(renderTexPtr);
    if (!s_Device || !s_StagingTex || !g_extMem || !dstRes || !g_lastTensor.defined())
    {
        FileLog("[DX12] PushLastTensorToUnity called before init", LOG_ERROR);
        return;
    }

    void* dstPtr = nullptr;
    cudaExternalMemoryBufferDesc bd = {0, g_totalBytes, 0};
    auto err = cudaExternalMemoryGetMappedBuffer(&dstPtr, g_extMem, &bd);
    if (err != cudaSuccess)
    {
        FileLog(std::string("[DX12] Map for write failed: ") + cudaGetErrorString(err), LOG_ERROR);
        return;
    }

    at::Tensor t = g_lastTensor;
    if (!t.is_contiguous()) t = t.contiguous();

    size_t rowBytes = (size_t)width * 4 * sizeof(float);
    err = cudaMemcpy2D(dstPtr, g_rowPitch, t.data_ptr(), rowBytes, rowBytes, height, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess)
    {
        FileLog(std::string("[DX12] cudaMemcpy2D failed: ") + cudaGetErrorString(err), LOG_ERROR);
        return;
    }

    FileLog("[DX12] Tensor → staging copy succeeded", LOG_VERBOSE);

    ID3D12GraphicsCommandList* cmdList = nullptr;
    HRESULT hr = s_Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, s_Allocator, nullptr, IID_PPV_ARGS(&cmdList));
    if (FAILED(hr) || !cmdList)
    {
        FileLog("[DX12] CreateCommandList failed: " + std::to_string(hr), LOG_ERROR);
        return;
    }

    D3D12_RESOURCE_BARRIER br[2] = {};
    br[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    br[0].Transition.pResource = s_StagingTex;
    br[0].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    br[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    br[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    br[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    br[1].Transition.pResource = dstRes;
    br[1].Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    br[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    br[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    cmdList->ResourceBarrier(2, br);
    cmdList->CopyResource(dstRes, s_StagingTex);
    std::swap(br[0].Transition.StateBefore, br[0].Transition.StateAfter);
    std::swap(br[1].Transition.StateBefore, br[1].Transition.StateAfter);
    cmdList->ResourceBarrier(2, br);
    cmdList->Close();

    s_GfxD3D12->ExecuteCommandList(cmdList, 0, nullptr);
    cmdList->Release();

    FileLog("[DX12] PushLastTensorToUnity completed", LOG_VERBOSE);
}

extern "C" HERMESARC_API const char* get_version()
{
    return "HermesArc version 0.90";
}