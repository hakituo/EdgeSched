#include "nvram_binding.h"
#include <iostream>
#include <chrono>

namespace ai_scheduler {

NVRAMContext& NVRAMContext::getInstance() {
    static NVRAMContext instance;
    return instance;
}

NVRAMContext::NVRAMContext() 
    : initialized_(false), 
#ifdef _WIN32
      hFile_(INVALID_HANDLE_VALUE), hMapFile_(NULL), 
#endif
      pBuf_(nullptr), pHeader_(nullptr), pBioState_(nullptr) {
}

NVRAMContext::~NVRAMContext() {
    shutdown();
}

bool NVRAMContext::initialize(const std::string& filePath) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_) return true;

    filePath_ = filePath;

#ifdef _WIN32
    // 1. Open File
    hFile_ = CreateFileA(
        filePath_.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        NULL,
        OPEN_EXISTING, // File must be created by Python side first usually, or we create it
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    if (hFile_ == INVALID_HANDLE_VALUE) {
        // Try creating if not exists
        hFile_ = CreateFileA(
            filePath_.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            NULL,
            OPEN_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            NULL
        );
        
        if (hFile_ == INVALID_HANDLE_VALUE) {
            std::cerr << "NVRAM: Failed to open file: " << filePath_ << std::endl;
            return false;
        }
    }

    // 2. Create File Mapping
    // 1MB size as defined in Python
    const DWORD FILE_SIZE = 1024 * 1024; 
    
    hMapFile_ = CreateFileMappingA(
        hFile_,
        NULL,
        PAGE_READWRITE,
        0,
        FILE_SIZE,
        "XiaoyouNVRAM" // Global name for IPC if needed, but file backed is enough
    );

    if (hMapFile_ == NULL) {
        std::cerr << "NVRAM: Could not create file mapping object." << std::endl;
        CloseHandle(hFile_);
        return false;
    }

    // 3. Map View
    pBuf_ = MapViewOfFile(
        hMapFile_,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        FILE_SIZE
    );

    if (pBuf_ == NULL) {
        std::cerr << "NVRAM: Could not map view of file." << std::endl;
        CloseHandle(hMapFile_);
        CloseHandle(hFile_);
        return false;
    }

    // 4. Setup Pointers
    // Layout MUST match nvram_layout.h and Python nvram_manager.py
    char* baseAddr = static_cast<char*>(pBuf_);
    pHeader_ = reinterpret_cast<NVRAMHeader*>(baseAddr + 0);
    pBioState_ = reinterpret_cast<BioState*>(baseAddr + 64); // BIO_OFFSET = 64

    // Verify Magic (optional, assuming Python initialized it)
    if (pHeader_->magic != NVRAM_MAGIC) {
        // If 0, maybe we initialized first. Set it.
        if (pHeader_->magic == 0) {
            pHeader_->magic = NVRAM_MAGIC;
            pHeader_->version = 1;
        }
    }

    initialized_ = true;
    std::cout << "NVRAM: C++ Context Initialized. Mapped to " << filePath_ << std::endl;
    return true;
#else
    // Linux implementation omitted for MVP Windows focus
    std::cerr << "NVRAM: Linux not implemented yet." << std::endl;
    return false;
#endif
}

void NVRAMContext::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) return;

#ifdef _WIN32
    if (pBuf_) UnmapViewOfFile(pBuf_);
    if (hMapFile_) CloseHandle(hMapFile_);
    if (hFile_ != INVALID_HANDLE_VALUE) CloseHandle(hFile_);
#endif

    initialized_ = false;
}

void NVRAMContext::updateBioState(double dopamine, double energy, double mood) {
    if (!initialized_ || !pBioState_) return;

    // Direct memory access - Zero Copy (relative to process memory)
    // No serialization needed
    pBioState_->dopamine = dopamine;
    pBioState_->energy = energy;
    pBioState_->mood = mood;
    
    // Update timestamp
    auto now = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = now.time_since_epoch();
    pBioState_->timestamp = diff.count();
}

BioState NVRAMContext::getBioState() {
    BioState state = {0};
    if (initialized_ && pBioState_) {
        state = *pBioState_;
    }
    return state;
}

void NVRAMContext::updateFlightRecorder(const std::string& status, uint32_t activeTaskId) {
    // Placeholder for flight recorder
    // We could use the unused space in TaskSlots or a dedicated area
    // For now, let's just log to console to simulate
    // In real implementation, we would write to offset 1024+...
}

} // namespace ai_scheduler
