#include "api/api_server.h"
#include "core/resource_isolation_scheduler.h"
#include "workers/cpu_tts_worker.h"
#include "workers/gpu_image_worker.h"
#include "workers/gpu_llm_worker.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <memory>

int main(int argc, char* argv[]) {
    int port = 8080;
    if (argc > 1) {
        try {
            port = std::stoi(argv[1]);
        } catch (...) {
            port = 8080;
        }
    }

    auto scheduler = std::make_shared<ai_scheduler::ResourceIsolationScheduler>();
    if (!scheduler->initialize(4)) {
        std::cerr << "Scheduler initialize failed" << std::endl;
        return 1;
    }

    auto ttsWorker = std::make_shared<ai_scheduler::CPUTTSWorker>("TTS_CPU_Worker", ai_scheduler::TTSEngineType::MOCK);
    auto llmWorker = std::make_shared<ai_scheduler::GPULLMWorker>("LLM_GPU_Worker");
    auto imgWorker = std::make_shared<ai_scheduler::GPUImageWorker>(1, ai_scheduler::ImageModel::SD15_TURBO);

    scheduler->addWorker(ttsWorker);
    scheduler->addWorker(llmWorker);
    scheduler->addWorker(imgWorker);

    auto server = std::make_shared<ai_scheduler::api::APIServer>(port);
    server->setScheduler(scheduler);
    server->setTTSWorker(ttsWorker);
    server->setLLMWorker(llmWorker);
    server->setImageWorker(imgWorker);

    if (!server->start()) {
        std::cerr << "API server start failed" << std::endl;
        scheduler->shutdown();
        return 1;
    }

    std::cout << "API server running on port: " << port << std::endl;
    std::cout << "Press Ctrl+C to exit" << std::endl;

    while (server->isRunning()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    scheduler->shutdown();
    return 0;
}
