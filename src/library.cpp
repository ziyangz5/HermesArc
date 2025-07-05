#include <fstream>
#include <string>
#include <vector>
#include <cuda_runtime_api.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <IUnityInterface.h>
#include <IUnityGraphics.h>
#include <IUnityGraphicsD3D12.h>
#include "library.h"
#include <c10/util/Exception.h>
#include <torch/serialize.h>

enum LogLevel {
    LOG_NONE = 0,
    LOG_ERROR = 1,
    LOG_VERBOSE = 2,
    LOG_DEBUG = 3
};

static int g_logVerbosity = LOG_ERROR;

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
static ID3D12Resource*        s_Backbuffers[4]  = {nullptr,nullptr,nullptr, nullptr};
static ID3D12Resource*        s_StagingTexs[4]   = {nullptr,nullptr,nullptr, nullptr};
static ID3D12Fence*           s_Fence           = nullptr;
static HANDLE                 s_FenceEvent      = nullptr;
static HANDLE                 g_shareHs[4]          = {nullptr,nullptr,nullptr, nullptr};
static cudaExternalMemory_t   g_extMems[4]      = {nullptr,nullptr,nullptr, nullptr};
static uint64_t               g_rowPitch        = 0;
static size_t                 g_totalBytes      = 0;
static FrameReadyCallback     g_FrameReadyCb    = nullptr;
static HANDLE                 g_WaitHandle      = nullptr;
static unsigned               g_lastWidth       = 0;
static unsigned               g_lastHeight      = 0;
static torch::Tensor          g_lastTensor[4];
static torch::jit::script::Module net;
static void noopCudaDeleter(void*) {}
static ID3D12Resource*        s_OutputStagingTex = nullptr;
static HANDLE                 g_OutputShareH     = nullptr;
static cudaExternalMemory_t   g_OutputExtMem     = nullptr;

static void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType);
extern "C" HERMESARC_API void CALLBACK FrameFenceCallback(PVOID context, BOOLEAN timedOut);
static uint64_t g_rowPitchArr[4]   = {0,0,0,0};
static uint64_t g_totalBytesArr[4] = {0,0,0,0};
static uint64_t g_outputRowPitch    = 0;
static uint64_t g_outputTotalBytes  = 0;
static ID3D12Resource* s_OutputStagingBuf = nullptr;
static HANDLE          g_outputShareH     = nullptr;
static cudaExternalMemory_t g_outputExtMem = nullptr;

struct TestNetImpl : torch::nn::Module {
    TestNetImpl() {
    }

    torch::Tensor forward(torch::Tensor x) {
        return x;
    }
};


TORCH_MODULE(TestNet);

static TestNet test_net;

extern "C" HERMESARC_API bool InitializeNeuralNetwork(const char* model_path)
{
    try {
        net = torch::jit::load(model_path);
        net.to(torch::kCUDA, torch::kFloat32);
        net.eval();
        test_net->to(torch::kCUDA, torch::kFloat32);
        test_net->eval();
    }
    catch (const c10::Error& e) {
        FileLog("Unable to load model.",LOG_ERROR);
        return false;
    }
    FileLog("[Torch] Model loaded.",LOG_VERBOSE);
    return true;
}


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

    if (s_Fence)        { s_Fence = nullptr; }
    if (s_FenceEvent)   { CloseHandle(s_FenceEvent); s_FenceEvent = nullptr; }
    if (g_WaitHandle)   { UnregisterWaitEx(g_WaitHandle, INVALID_HANDLE_VALUE); g_WaitHandle = nullptr; }
    for (int i = 0;i<4;i++)
    {
        if (g_extMems[i])       { cudaDestroyExternalMemory(g_extMems[i]); }
        if (g_shareHs[i])       { CloseHandle(g_shareHs[i]); }
        if (s_StagingTexs[i])   { s_StagingTexs[i]->Release();   s_StagingTexs[i] = nullptr; }
    }

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

extern "C" HERMESARC_API void CaptureFrameAsync(
        void* ptr0, void* ptr1, void* ptr2, void* ptr3,
        unsigned width, unsigned height)
{
    
    s_Backbuffers[0] = reinterpret_cast<ID3D12Resource*>(ptr0);
    s_Backbuffers[1] = reinterpret_cast<ID3D12Resource*>(ptr1);
    s_Backbuffers[2] = reinterpret_cast<ID3D12Resource*>(ptr2);
    s_Backbuffers[3] = reinterpret_cast<ID3D12Resource*>(ptr3);

    
    if (!s_Allocator && s_Device)
    {
        HRESULT hr = s_Device->CreateCommandAllocator(
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                IID_PPV_ARGS(&s_Allocator));
        if (FAILED(hr))
        {
            FileLog("[DX12] CreateCommandAllocator failed: " + std::to_string(hr), LOG_ERROR);
            return;
        }
    }

    
    for (int i = 0; i < 4; ++i)
    {
        if (!s_StagingTexs[i] && s_Device)
        {
            
            D3D12_RESOURCE_DESC texDesc = s_Backbuffers[i]->GetDesc();
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout = {};
            UINT numRows = 0;
            UINT64 rowBytes = 0, totalBytes = 0;
            s_Device->GetCopyableFootprints(
                    &texDesc, 0, 1, 0,
                    &layout, &numRows, &rowBytes, &totalBytes);

            
            g_rowPitchArr[i]   = layout.Footprint.RowPitch;
            g_totalBytesArr[i] = totalBytes;

            
            D3D12_RESOURCE_DESC bufDesc = {};
            bufDesc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
            bufDesc.Alignment        = 0;
            bufDesc.Width            = totalBytes;
            bufDesc.Height           = 1;
            bufDesc.DepthOrArraySize = 1;
            bufDesc.MipLevels        = 1;
            bufDesc.Format           = DXGI_FORMAT_UNKNOWN;
            bufDesc.SampleDesc.Count = 1;
            bufDesc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            bufDesc.Flags            = D3D12_RESOURCE_FLAG_NONE;

            D3D12_HEAP_PROPERTIES hp = {};
            hp.Type = D3D12_HEAP_TYPE_DEFAULT;

            HRESULT hr = s_Device->CreateCommittedResource(
                    &hp,
                    D3D12_HEAP_FLAG_SHARED,
                    &bufDesc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    nullptr,
                    IID_PPV_ARGS(&s_StagingTexs[i]));
            if (FAILED(hr))
            {
                FileLog("[DX12] CreateCommittedResource (buffer) failed: " + std::to_string(hr), LOG_ERROR);
                return;
            }

            
            hr = s_Device->CreateSharedHandle(
                    s_StagingTexs[i],
                    nullptr,
                    GENERIC_ALL,
                    nullptr,
                    &g_shareHs[i]);
            if (FAILED(hr))
            {
                FileLog("[DX12] CreateSharedHandle failed: " + std::to_string(hr), LOG_ERROR);
                return;
            }

            
            cudaExternalMemoryHandleDesc memDesc = {};
            memDesc.type                = cudaExternalMemoryHandleTypeD3D12Resource;
            memDesc.handle.win32.handle = g_shareHs[i];
            memDesc.size                = totalBytes;
            memDesc.flags               = cudaExternalMemoryDedicated;
            cudaError_t impErr = cudaImportExternalMemory(&g_extMems[i], &memDesc);
            FileLog(std::string("cudaImportExternalMemory -> ") + cudaGetErrorString(impErr), LOG_VERBOSE);
            if (impErr != cudaSuccess)
                return;
        }
    }

    
    s_Allocator->Reset();
    ID3D12GraphicsCommandList* cmdList = nullptr;
    HRESULT hrCL = s_Device->CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            s_Allocator,
            nullptr,
            IID_PPV_ARGS(&cmdList));
    if (FAILED(hrCL) || !cmdList)
    {
        FileLog("[DX12] CreateCommandList failed: " + std::to_string(hrCL), LOG_ERROR);
        return;
    }

    for (int i = 0; i < 4; ++i)
    {
        
        D3D12_RESOURCE_BARRIER br = {};
        br.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        br.Transition.pResource   = s_Backbuffers[i];
        br.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        br.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_SOURCE;
        br.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cmdList->ResourceBarrier(1, &br);

        
        D3D12_RESOURCE_DESC texDesc = s_Backbuffers[i]->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout = {};
        UINT numRows = 0;
        UINT64 rowBytes = 0, totalBytes = 0;
        s_Device->GetCopyableFootprints(
                &texDesc, 0, 1, 0,
                &layout, &numRows, &rowBytes, &totalBytes);

        
        D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
        srcLoc.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        srcLoc.pResource        = s_Backbuffers[i];
        srcLoc.SubresourceIndex = 0;

        D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
        dstLoc.Type             = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dstLoc.pResource        = s_StagingTexs[i];
        dstLoc.PlacedFootprint  = layout;

        cmdList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);

        
        std::swap(br.Transition.StateBefore, br.Transition.StateAfter);
        cmdList->ResourceBarrier(1, &br);
    }

    cmdList->Close();
    UINT64 fenceValue = s_GfxD3D12->ExecuteCommandList(cmdList, 0, nullptr);
    cmdList->Release();

    s_Fence->SetEventOnCompletion(fenceValue, s_FenceEvent);
    RegisterWaitForSingleObject(
            &g_WaitHandle,
            s_FenceEvent,
            FrameFenceCallback,
            nullptr,
            INFINITE,
            WT_EXECUTEONLYONCE);

    g_lastWidth  = width;
    g_lastHeight = height;
    FileLog("[DX12] CaptureFrameAsync scheduled callback.", LOG_VERBOSE);
}


extern "C" HERMESARC_API void CALLBACK FrameFenceCallback(PVOID /*context*/, BOOLEAN /*timedOut*/)
{
    FileLog(">>> FrameFenceCallback start", LOG_VERBOSE);

    for (int i = 0; i < 4; ++i)
    {
        
        void* devPtr = nullptr;
        cudaExternalMemoryBufferDesc bufDesc = {};
        bufDesc.offset = 0;
        bufDesc.size   = g_totalBytesArr[i];
        bufDesc.flags  = 0;

        cudaError_t mapErr = cudaExternalMemoryGetMappedBuffer(
                &devPtr, g_extMems[i], &bufDesc);
        FileLog(std::string("  cudaExternalMemoryGetMappedBuffer -> ")
                + cudaGetErrorString(mapErr),
                LOG_VERBOSE);
        if (mapErr != cudaSuccess)
            return;

        
        int64_t H = static_cast<int64_t>(g_lastHeight);
        int64_t W = static_cast<int64_t>(g_lastWidth);
        int64_t C = 4;
        int64_t rowPitchElems = static_cast<int64_t>(g_rowPitchArr[i] / sizeof(float));

        std::vector<int64_t> dims    = { H, W, C };
        std::vector<int64_t> strides = { rowPitchElems, C, 1 };

        try
        {
            auto t1 = torch::from_blob(
                    devPtr,
                    dims,
                    strides,
                    &noopCudaDeleter,
                    torch::TensorOptions()
                            .dtype(torch::kFloat32)
                            .device(torch::kCUDA, 0)
            );
            auto t2 = t1.clone();
            g_lastTensor[i] = t2;

            
            float meanVal = t2.mean().item<float>();
            FileLog("  mean() = " + std::to_string(meanVal), LOG_VERBOSE);
            if (g_FrameReadyCb)
                g_FrameReadyCb(meanVal);
        }
        catch (const std::exception& e)
        {
            FileLog(std::string("  EXCEPTION: ") + e.what(), LOG_ERROR);
        }
    }
}

extern "C" HERMESARC_API void PushLastTensorToUnity(void* renderTexPtr, unsigned width, unsigned height)
{
    ID3D12Resource* dstRes = reinterpret_cast<ID3D12Resource*>(renderTexPtr);
    if (!s_Device || !dstRes ||
        !g_lastTensor[0].defined() ||
        !g_lastTensor[1].defined() ||
        !g_lastTensor[2].defined() ||
        !g_lastTensor[3].defined())
    {
        FileLog("[DX12] PushLastTensorToUnity called before init", LOG_ERROR);
        return;
    }

    
    if (!s_OutputStagingBuf)
    {
        
        D3D12_RESOURCE_DESC dstDesc = dstRes->GetDesc();

        
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout = {};
        UINT numRows = 0;
        UINT64 rowBytes = 0, totalBytes = 0;
        s_Device->GetCopyableFootprints(
                &dstDesc,        
                0, 1, 0,
                &layout, &numRows, &rowBytes, &totalBytes);

        g_outputRowPitch   = layout.Footprint.RowPitch;
        g_outputTotalBytes = totalBytes;
        
        D3D12_RESOURCE_DESC bufDesc = {};
        bufDesc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
        bufDesc.Alignment        = 0;
        bufDesc.Width            = totalBytes;
        bufDesc.Height           = 1;
        bufDesc.DepthOrArraySize = 1;
        bufDesc.MipLevels        = 1;
        bufDesc.Format           = DXGI_FORMAT_UNKNOWN;
        bufDesc.SampleDesc.Count = 1;
        bufDesc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        bufDesc.Flags            = D3D12_RESOURCE_FLAG_NONE;

        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;

        HRESULT hr = s_Device->CreateCommittedResource(
                &hp,
                D3D12_HEAP_FLAG_SHARED,
                &bufDesc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr,
                IID_PPV_ARGS(&s_OutputStagingBuf));
        if (FAILED(hr)) {
            FileLog("[DX12] CreateCommittedResource (output buffer) failed: " + std::to_string(hr), LOG_ERROR);
            return;
        }
        
        hr = s_Device->CreateSharedHandle(
                s_OutputStagingBuf,
                nullptr,
                GENERIC_ALL,
                nullptr,
                &g_outputShareH);
        if (FAILED(hr)) {
            FileLog("[DX12] CreateSharedHandle (output) failed: " + std::to_string(hr), LOG_ERROR);
            return;
        }

        cudaExternalMemoryHandleDesc memDesc{};
        memDesc.type                = cudaExternalMemoryHandleTypeD3D12Resource;
        memDesc.handle.win32.handle = g_outputShareH;
        memDesc.size                = totalBytes;
        memDesc.flags               = cudaExternalMemoryDedicated;
        cudaError_t impErr = cudaImportExternalMemory(&g_outputExtMem, &memDesc);
        if (impErr != cudaSuccess) {
            FileLog(std::string("[DX12] cudaImportExternalMemory (output) -> ") + cudaGetErrorString(impErr), LOG_ERROR);
            return;
        }

        FileLog("[DX12] Allocated output ROW-MAJOR buffer + imported into CUDA", LOG_VERBOSE);
    }

    at::Tensor normal = g_lastTensor[0].contiguous();
    at::Tensor depth  = g_lastTensor[1].contiguous();
    at::Tensor tex    = g_lastTensor[2].contiguous();
    at::Tensor render = torch::pow(g_lastTensor[3].contiguous(),1/2.2f);

    auto normal_c = normal.slice(2, 0, 3);
    auto depth_r  = depth.select(2, 0).unsqueeze(2);
    auto tex_c    = tex.slice(2, 0, 3);
    auto render_c = render.slice(2, 0, 3);

    at::Tensor input_hw_c = torch::cat({ depth_r, normal_c, render_c, tex_c }, 2).contiguous();
    at::Tensor input = input_hw_c.permute({2,0,1}).unsqueeze(0).contiguous();


    at::Tensor result;
    try {
        result = net.forward({input.flip({2})}).toTensor().flip({2});
        result = result.clamp_(0.0, 1.0).nan_to_num_(0.0, 0.0, 0.0);
    } catch (const c10::Error& e) {
        FileLog(std::string("Torch C++ error in forward(): ") + e.what(), LOG_ERROR);
        return;
    }

    if (g_logVerbosity == LOG_DEBUG)
    {
        at::Tensor input_to_save = input.to(at::kCPU);
        torch::save(input_to_save, "input.pth");
        at::Tensor output_to_save = result.to(at::kCPU);
        torch::save(output_to_save, "output.pth");
    }
    
    result = result.squeeze(0).permute({1,2,0}).contiguous();
    {
        int64_t H = result.size(0), W = result.size(1);
        at::Tensor ones = torch::ones({H,W,1}, result.options());
        result = torch::cat({ result, ones }, 2);
    }
    torch::cuda::synchronize();
    
    void* dstPtr = nullptr;
    cudaExternalMemoryBufferDesc bufDesc{};
    bufDesc.offset = 0;
    bufDesc.size   = g_outputTotalBytes;
    bufDesc.flags  = 0;

    cudaError_t mapErr = cudaExternalMemoryGetMappedBuffer(&dstPtr, g_outputExtMem, &bufDesc);
    if (mapErr != cudaSuccess) {
        FileLog(std::string("[DX12] cudaExternalMemoryGetMappedBuffer (output) -> ") + cudaGetErrorString(mapErr), LOG_ERROR);
        return;
    }

    size_t srcPitchBytes = width * 4 * sizeof(float);
    size_t dstPitchBytes = g_outputRowPitch;
    size_t copyHeight    = height;

    cudaError_t memcpyErr = cudaMemcpy2D(
            dstPtr,
            dstPitchBytes,
            result.data_ptr(),
            srcPitchBytes,
            srcPitchBytes,
            copyHeight,
            cudaMemcpyDeviceToDevice);
    if (memcpyErr != cudaSuccess) {
        FileLog(std::string("[DX12] cudaMemcpy2D (output) failed: ") + cudaGetErrorString(memcpyErr), LOG_ERROR);
        return;
    }
    torch::cuda::synchronize();

    FileLog("[DX12] Copied NN result into output buffer", LOG_VERBOSE);

    
    s_Allocator->Reset();
    ID3D12GraphicsCommandList* cmdList = nullptr;
    HRESULT hrCL = s_Device->CreateCommandList(
            0, D3D12_COMMAND_LIST_TYPE_DIRECT,
            s_Allocator, nullptr,
            IID_PPV_ARGS(&cmdList));
    if (FAILED(hrCL) || !cmdList) {
        FileLog("[DX12] CreateCommandList failed (output copy): " + std::to_string(hrCL), LOG_ERROR);
        return;
    }

    
    D3D12_RESOURCE_BARRIER br[2] = {};
    br[0].Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    br[0].Transition.pResource   = s_OutputStagingBuf;
    br[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    br[0].Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_SOURCE;
    br[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    br[1].Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    br[1].Transition.pResource   = dstRes;
    br[1].Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    br[1].Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_DEST;
    br[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    cmdList->ResourceBarrier(2, br);

    
    D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
    srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    srcLoc.pResource = s_OutputStagingBuf;

    
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout2 = {};
    UINT rows2 = 0;
    UINT64 rb2 = 0, tb2 = 0;
    {
        D3D12_RESOURCE_DESC dstDesc2 = dstRes->GetDesc();
        s_Device->GetCopyableFootprints(&dstDesc2, 0, 1, 0, &layout2, &rows2, &rb2, &tb2);
    }
    srcLoc.PlacedFootprint = layout2;

    D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
    dstLoc.Type            = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dstLoc.pResource       = dstRes;
    dstLoc.SubresourceIndex = 0;

    
    cmdList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);

    
    std::swap(br[0].Transition.StateBefore, br[0].Transition.StateAfter);
    std::swap(br[1].Transition.StateBefore, br[1].Transition.StateAfter);
    cmdList->ResourceBarrier(2, br);

    cmdList->Close();
    s_GfxD3D12->ExecuteCommandList(cmdList, 0, nullptr);
    cmdList->Release();

    FileLog("[DX12] PushLastTensorToUnity output copy complete", LOG_VERBOSE);
}

extern "C" HERMESARC_API const char* get_version()
{
    return "HermesArc version 0.93c.dev3";
}