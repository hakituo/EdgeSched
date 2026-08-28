#ifndef NVRAM_BINDING_H
#define NVRAM_BINDING_H

#include <string>
#include <memory>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#else
// Linux simulation or placeholder if needed later
#endif

#include "../../include/nvram_layout.h"

namespace ai_scheduler {

class NVRAMContext {
public:
    static NVRAMContext& getInstance();

    bool initialize(const std::string& filePath);
    void shutdown();
    
    // Write accessors
    void updateBioState(double dopamine, double energy, double mood);
    void updateFlightRecorder(const std::string& status, uint32_t activeTaskId);

    // Read accessors
    BioState getBioState();

private:
    NVRAMContext();
    ~NVRAMContext();
    NVRAMContext(const NVRAMContext&) = delete;
    NVRAMContext& operator=(const NVRAMContext&) = delete;

    std::string filePath_;
    bool initialized_;
    std::mutex mutex_;

#ifdef _WIN32
    HANDLE hFile_;
    HANDLE hMapFile_;
    LPVOID pBuf_;
#else
    void* pBuf_;
    int fd_;
#endif
    
    // Pointers to mapped regions
    NVRAMHeader* pHeader_;
    BioState* pBioState_;
    // Flight Recorder Area (simulated at end of BioState or specifically offset)
    // We will use a reserved area if not in layout, or extend layout.
    // For now, let's assume we strictly follow NVRAMLayout.
};

} // namespace ai_scheduler

#endif // NVRAM_BINDING_H
