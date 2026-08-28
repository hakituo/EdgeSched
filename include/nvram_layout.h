#ifndef NVRAM_LAYOUT_H
#define NVRAM_LAYOUT_H

#include <cstdint>
#include <ctime>

// Magic Number to verify file format (corresponds to Python side)
#define NVRAM_MAGIC 0x58594E56 // "XYNV" in hex

// Use 1-byte packing to match Python's struct.pack default behavior for standard types
// though Python's standard size usually aligns well, explicit packing is safer for cross-language
#pragma pack(push, 1)

// Header Structure (Offset 0x00)
struct NVRAMHeader {
    uint32_t magic;      // 4 bytes
    uint32_t version;    // 4 bytes
    // Total: 8 bytes
};

// Bio System State (Offset 0x08)
// Python format: "ddd d" (4 doubles)
struct BioState {
    double dopamine;     // 8 bytes
    double energy;       // 8 bytes
    double mood;         // 8 bytes
    double timestamp;    // 8 bytes
    // Total: 32 bytes
};

// Task Slot Definition
// Python format: "32s i i d"
struct TaskSlot {
    char task_id[32];    // 32 bytes (Fixed width string)
    int32_t priority;    // 4 bytes
    int32_t status;      // 4 bytes (0=Empty, 1=Pending, 2=Running, 3=Completed, 4=Failed)
    double timestamp;    // 8 bytes
    // Total: 48 bytes
};

// The complete memory map layout
struct NVRAMLayout {
    NVRAMHeader header;              // 0
    BioState bio_state;              // 8
    TaskSlot tasks[32];              // 8 + 32 = 40.  40 + (32 * 48) = 1576 bytes
    // Remaining space is padded or used for KV Swap simulation
};

#pragma pack(pop)

// Constants for C++ Scheduler
const int MAX_TASKS = 32;
const int STATUS_EMPTY = 0;
const int STATUS_PENDING = 1;
const int STATUS_RUNNING = 2;
const int STATUS_COMPLETED = 3;
const int STATUS_FAILED = 4;
const int STATUS_CANCELLED = 5;

#endif // NVRAM_LAYOUT_H
