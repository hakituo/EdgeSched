#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>
#include <memory>

struct FlightLogEntry {
    std::string category;
    std::string message;
    std::chrono::steady_clock::time_point timestamp;
};

class FlightRecorder {
public:
    static FlightRecorder& getInstance() {
        static FlightRecorder instance;
        return instance;
    }

    void log(const std::string& category, const std::string& message) {
        std::lock_guard<std::mutex> lock(mutex_);
        FlightLogEntry entry;
        entry.category = category;
        entry.message = message;
        entry.timestamp = std::chrono::steady_clock::now();
        entries_.push_back(std::move(entry));

        if (entries_.size() > max_entries_) {
            entries_.erase(entries_.begin(), entries_.begin() + (entries_.size() - max_entries_));
        }
    }

    const std::vector<FlightLogEntry>& getEntries() const {
        return entries_;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        entries_.clear();
    }

    bool dumpToFile(const std::string& filepath) const {
        std::ofstream ofs(filepath, std::ios::app);
        if (!ofs.is_open()) return false;
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& e : entries_) {
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                e.timestamp.time_since_epoch()).count();
            ofs << "[" << ms << "] " << e.category << ": " << e.message << "\n";
        }
        return true;
    }

private:
    FlightRecorder() = default;
    FlightRecorder(const FlightRecorder&) = delete;
    FlightRecorder& operator=(const FlightRecorder&) = delete;

    mutable std::mutex mutex_;
    std::vector<FlightLogEntry> entries_;
    size_t max_entries_ = 10000;
};
