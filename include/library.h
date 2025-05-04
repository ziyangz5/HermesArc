#ifndef HERMESARC_LIBRARY_H
#define HERMESARC_LIBRARY_H

#ifdef HERMESARC_EXPORTS
#define HERMESARC_API __declspec(dllexport)
#else
#define HERMESARC_API __declspec(dllimport)
#endif

extern "C" HERMESARC_API const char* get_version();

extern "C" HERMESARC_API void InitializeDX12Interop(void* backbufferPtr, unsigned width, unsigned height);

extern "C" HERMESARC_API float CaptureFrameToTensor(unsigned width, unsigned height);

#endif // HERMESARC_LIBRARY_H