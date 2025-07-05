#ifndef HERMESARC_LIBRARY_H
#define HERMESARC_LIBRARY_H

#ifdef HERMESARC_EXPORTS
#define HERMESARC_API __declspec(dllexport)
#else
#define HERMESARC_API __declspec(dllimport)
#endif

extern "C" HERMESARC_API const char* get_version();

extern "C" HERMESARC_API bool InitializeNeuralNetwork(const char* model_path);

extern "C" HERMESARC_API void CaptureFrameAsync(void* ptr0, void* ptr1, void* ptr2, void* ptr3,unsigned width, unsigned height);

#endif // HERMESARC_LIBRARY_H