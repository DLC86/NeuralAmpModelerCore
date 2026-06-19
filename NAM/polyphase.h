#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "dsp.h"
#include "slimmable.h"

#if defined(_WIN32)
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
  #include <avrt.h>
#endif

#if defined(__APPLE__)
  #include <pthread.h>
  #if __has_include(<pthread/qos.h>)
    #include <pthread/qos.h>
  #endif
  #if __has_include(<os/workgroup.h>)
    #include <os/workgroup.h>
    #include <os/object.h>
    #define NAM_POLYPHASE_HAS_AUDIO_WORKGROUP 1
  #else
    #define NAM_POLYPHASE_HAS_AUDIO_WORKGROUP 0
  #endif
#else
  #define NAM_POLYPHASE_HAS_AUDIO_WORKGROUP 0
#endif

namespace nam
{

/// A model-rate DSP that decomposes an integer-oversampled WaveNet stream into
/// independent native-rate phases. The object owns all phase models, buffers,
/// and persistent workers, so the plug-in wrapper sees one ordinary DSP.
class PolyphaseOversampledDSP final : public DSP, public SlimmableModel
{
public:
  PolyphaseOversampledDSP(std::vector<std::unique_ptr<DSP>> phases, double expectedSampleRate, int threads)
  : DSP(1, 1, expectedSampleRate)
  , mPhases(std::move(phases))
  {
    const int factor = GetFactor();
    if (factor < 2)
      throw std::invalid_argument("PolyphaseOversampledDSP requires at least two phases.");

    for (const auto& phase : mPhases)
    {
      if (!phase || phase->NumInputChannels() != 1 || phase->NumOutputChannels() != 1)
        throw std::invalid_argument("PolyphaseOversampledDSP phases must be non-null mono DSPs.");
    }

    mThreads = std::clamp(threads, 1, factor);
    mInputBuffers.resize(static_cast<size_t>(factor));
    mOutputBuffers.resize(static_cast<size_t>(factor));
    mPhaseFrames.assign(static_cast<size_t>(factor), 0);
    mOutputCursors.assign(static_cast<size_t>(factor), 0);

    if (mPhases[0]->HasLoudness())
      SetLoudness(mPhases[0]->GetLoudness());
    if (mPhases[0]->HasInputLevel())
      SetInputLevel(mPhases[0]->GetInputLevel());
    if (mPhases[0]->HasOutputLevel())
      SetOutputLevel(mPhases[0]->GetOutputLevel());

    const int workerCount = mThreads - 1;
    mWorkers.reserve(static_cast<size_t>(workerCount));
    for (int worker = 0; worker < workerCount; worker++)
      mWorkers.emplace_back([this, worker] { WorkerLoop(worker); });
  }

  ~PolyphaseOversampledDSP() override
  {
    mStop.store(true, std::memory_order_release);
    mGeneration.fetch_add(1, std::memory_order_release);
    mSleepCV.notify_all();
    for (auto& worker : mWorkers)
      if (worker.joinable())
        worker.join();

#if NAM_POLYPHASE_HAS_AUDIO_WORKGROUP
    if (mAudioWorkgroup != nullptr)
      os_release(mAudioWorkgroup);
#endif
  }

  int GetFactor() const { return static_cast<int>(mPhases.size()); }
  int GetThreads() const { return mThreads; }

  void SetAudioWorkgroup(void* workgroup)
  {
#if NAM_POLYPHASE_HAS_AUDIO_WORKGROUP
    std::lock_guard<std::mutex> lock(mAudioWorkgroupMutex);
    os_workgroup_t newWorkgroup = static_cast<os_workgroup_t>(workgroup);
    if (newWorkgroup == mAudioWorkgroup)
      return;

    if (newWorkgroup != nullptr)
      os_retain(newWorkgroup);
    if (mAudioWorkgroup != nullptr)
      os_release(mAudioWorkgroup);
    mAudioWorkgroup = newWorkgroup;
#else
    (void)workgroup;
#endif
  }

  void Reset(double sampleRate, int maxBufferSize) override
  {
    mExternalSampleRate = sampleRate;
    mHaveExternalSampleRate = true;
    SetMaxBufferSize(maxBufferSize);
    mPhaseOffset = 0;

    const int factor = GetFactor();
    const double phaseRate = sampleRate / static_cast<double>(factor);
    const int phaseBufferSize = (maxBufferSize + factor - 1) / factor + 1;
    for (auto& phase : mPhases)
      phase->ResetAndPrewarm(phaseRate, phaseBufferSize);
  }

  void prewarm() override
  {
    for (auto& phase : mPhases)
      phase->prewarm();
  }

  void SetSlimmableSize(double value) override
  {
    for (auto& phase : mPhases)
      if (auto* slimmable = dynamic_cast<SlimmableModel*>(phase.get()))
        slimmable->SetSlimmableSize(value);
  }

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, int numFrames) override
  {
    if (numFrames <= 0)
      return;
    if (numFrames > GetMaxBufferSize())
      SetMaxBufferSize(numFrames);

    const int factor = GetFactor();
    std::fill(mPhaseFrames.begin(), mPhaseFrames.end(), 0);

    int phaseIndex = mPhaseOffset;
    for (int frame = 0; frame < numFrames; frame++)
    {
      auto& phaseInput = mInputBuffers[static_cast<size_t>(phaseIndex)];
      int& phaseFrames = mPhaseFrames[static_cast<size_t>(phaseIndex)];
      phaseInput[static_cast<size_t>(phaseFrames++)] = input[0][frame];
      if (++phaseIndex == factor)
        phaseIndex = 0;
    }

    if (mThreads == 1)
    {
      for (int phase = 0; phase < factor; phase++)
        ProcessPhase(phase);
    }
    else
    {
      mCompletedWorkers.store(0, std::memory_order_relaxed);
      mGeneration.fetch_add(1, std::memory_order_release);
      mSleepCV.notify_all();

      for (int phase = 0; phase < factor; phase += mThreads)
        ProcessPhase(phase);

      int spins = 0;
      const int workerCount = mThreads - 1;
      while (mCompletedWorkers.load(std::memory_order_acquire) < workerCount)
      {
        PauseRealtimeThread();
        if (++spins >= 16384)
        {
          spins = 0;
          std::this_thread::yield();
        }
      }
    }

    std::fill(mOutputCursors.begin(), mOutputCursors.end(), 0);
    phaseIndex = mPhaseOffset;
    for (int frame = 0; frame < numFrames; frame++)
    {
      auto& cursor = mOutputCursors[static_cast<size_t>(phaseIndex)];
      output[0][frame] = mOutputBuffers[static_cast<size_t>(phaseIndex)][static_cast<size_t>(cursor++)];
      if (++phaseIndex == factor)
        phaseIndex = 0;
    }

    mPhaseOffset = (mPhaseOffset + numFrames) % factor;
  }

protected:
  void SetMaxBufferSize(int maxBufferSize) override
  {
    DSP::SetMaxBufferSize(maxBufferSize);
    const int phaseCapacity = (maxBufferSize + GetFactor() - 1) / GetFactor() + 1;
    for (int phase = 0; phase < GetFactor(); phase++)
    {
      mInputBuffers[static_cast<size_t>(phase)].resize(static_cast<size_t>(phaseCapacity));
      mOutputBuffers[static_cast<size_t>(phase)].resize(static_cast<size_t>(phaseCapacity));
    }
  }

private:
  static void PauseRealtimeThread()
  {
#if defined(__arm64__) || defined(__aarch64__)
    __asm__ __volatile__("yield");
#elif defined(_WIN32)
    YieldProcessor();
#else
    std::this_thread::yield();
#endif
  }

  static void ConfigureWorkerThread()
  {
#if defined(_WIN32)
    DWORD taskIndex = 0;
    HANDLE mmcss = AvSetMmThreadCharacteristicsA("Pro Audio", &taskIndex);
    if (mmcss != nullptr)
      AvSetMmThreadPriority(mmcss, AVRT_PRIORITY_HIGH);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#elif defined(__APPLE__) && defined(QOS_CLASS_USER_INTERACTIVE)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
  }

  void ProcessPhase(int phase)
  {
    const int frames = mPhaseFrames[static_cast<size_t>(phase)];
    if (frames <= 0)
      return;
    NAM_SAMPLE* input = mInputBuffers[static_cast<size_t>(phase)].data();
    NAM_SAMPLE* output = mOutputBuffers[static_cast<size_t>(phase)].data();
    mPhases[static_cast<size_t>(phase)]->process(&input, &output, frames);
  }

  void WorkerLoop(int worker)
  {
    ConfigureWorkerThread();
    unsigned seenGeneration = 0;
#if defined(__APPLE__) && (defined(__arm64__) || defined(__aarch64__))
    int idleSpins = 0;
#endif

    for (;;)
    {
      const unsigned generation = mGeneration.load(std::memory_order_acquire);
      if (generation == seenGeneration)
      {
        if (mStop.load(std::memory_order_acquire))
          return;

#if defined(__APPLE__) && (defined(__arm64__) || defined(__aarch64__))
        // Apple Silicon benefits from staying runnable between tiny adjacent
        // audio blocks. Do not use this on Windows: MMCSS high-priority
        // workers would otherwise consume every logical CPU while idle.
        if (idleSpins++ < 262144)
        {
          PauseRealtimeThread();
          continue;
        }

        std::unique_lock<std::mutex> lock(mSleepMutex);
        mSleepCV.wait_for(lock, std::chrono::microseconds(500), [this, seenGeneration] {
          return mStop.load(std::memory_order_acquire)
                 || mGeneration.load(std::memory_order_acquire) != seenGeneration;
        });
        idleSpins = 0;
#else
        std::unique_lock<std::mutex> lock(mSleepMutex);
        mSleepCV.wait(lock, [this, seenGeneration] {
          return mStop.load(std::memory_order_acquire)
                 || mGeneration.load(std::memory_order_acquire) != seenGeneration;
        });
#endif
        continue;
      }

      seenGeneration = generation;
#if defined(__APPLE__) && (defined(__arm64__) || defined(__aarch64__))
      idleSpins = 0;
#endif
      if (mStop.load(std::memory_order_acquire))
        return;

#if NAM_POLYPHASE_HAS_AUDIO_WORKGROUP
      os_workgroup_t audioWorkgroup = nullptr;
      {
        std::lock_guard<std::mutex> lock(mAudioWorkgroupMutex);
        audioWorkgroup = mAudioWorkgroup;
      }
      os_workgroup_join_token_s workgroupToken {};
      bool joinedWorkgroup = false;
      if (audioWorkgroup != nullptr)
      {
        if (__builtin_available(macOS 11.0, iOS 14.0, *))
          joinedWorkgroup = os_workgroup_join(audioWorkgroup, &workgroupToken) == 0;
      }
#endif

      for (int phase = worker + 1; phase < GetFactor(); phase += mThreads)
        ProcessPhase(phase);

#if NAM_POLYPHASE_HAS_AUDIO_WORKGROUP
      if (joinedWorkgroup)
        os_workgroup_leave(audioWorkgroup, &workgroupToken);
#endif

      mCompletedWorkers.fetch_add(1, std::memory_order_release);
    }
  }

  std::vector<std::unique_ptr<DSP>> mPhases;
  int mThreads = 1;
  int mPhaseOffset = 0;
  std::vector<std::vector<NAM_SAMPLE>> mInputBuffers;
  std::vector<std::vector<NAM_SAMPLE>> mOutputBuffers;
  std::vector<int> mPhaseFrames;
  std::vector<int> mOutputCursors;

  std::vector<std::thread> mWorkers;
  std::atomic<unsigned> mGeneration {0};
  std::atomic<int> mCompletedWorkers {0};
  std::atomic<bool> mStop {false};
  std::mutex mSleepMutex;
  std::condition_variable mSleepCV;
#if NAM_POLYPHASE_HAS_AUDIO_WORKGROUP
  std::mutex mAudioWorkgroupMutex;
  os_workgroup_t mAudioWorkgroup = nullptr;
#endif
};

} // namespace nam

#undef NAM_POLYPHASE_HAS_AUDIO_WORKGROUP
