#include "cpu_pool.h"
#include "cpu_threads.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <thread>

namespace orbitquant::cpu {
namespace {

struct Job {
  std::vector<std::pair<std::int64_t, std::int64_t>> const *ranges;
  std::function<void(std::int64_t, std::int64_t)> const *fn;
  std::atomic<std::size_t> next_range{0};
  std::atomic<int> pending_workers{0};
};

class WorkerPool {
 public:
  static WorkerPool &instance() {
    // Intentionally leaked: joining workers during static destruction can
    // deadlock interpreter teardown, so the pool lives for the process.
    static WorkerPool *pool = new WorkerPool(requested_threads() - 1);
    return *pool;
  }

  int worker_count() const {
    return worker_count_;
  }

  void run(Job &job) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      job.pending_workers.store(worker_count_, std::memory_order_relaxed);
      job_ = &job;
      ++generation_;
    }
    wake_cv_.notify_all();
    drain(job);
    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [&job] {
      return job.pending_workers.load(std::memory_order_acquire) == 0;
    });
    job_ = nullptr;
  }

 private:
  explicit WorkerPool(int requested_workers)
      : worker_count_(requested_workers > 0 ? requested_workers : 0) {
    for (int worker = 0; worker < worker_count_; ++worker) {
      std::thread([this] { worker_loop(); }).detach();
    }
  }

  static void drain(Job &job) {
    const std::size_t range_count = job.ranges->size();
    for (;;) {
      const std::size_t index =
          job.next_range.fetch_add(1, std::memory_order_relaxed);
      if (index >= range_count) {
        return;
      }
      auto const &range = (*job.ranges)[index];
      (*job.fn)(range.first, range.second);
    }
  }

  void worker_loop() {
    std::uint64_t seen_generation = 0;
    for (;;) {
      Job *job = nullptr;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        wake_cv_.wait(lock, [this, seen_generation] {
          return generation_ != seen_generation;
        });
        seen_generation = generation_;
        job = job_;
      }
      if (job == nullptr) {
        continue;
      }
      drain(*job);
      if (job->pending_workers.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard<std::mutex> lock(mutex_);
        done_cv_.notify_all();
      }
    }
  }

  const int worker_count_;
  std::mutex mutex_;
  std::condition_variable wake_cv_;
  std::condition_variable done_cv_;
  Job *job_ = nullptr;
  std::uint64_t generation_ = 0;
};

}  // namespace

void run_ranges(
    std::vector<std::pair<std::int64_t, std::int64_t>> const &ranges,
    std::function<void(std::int64_t, std::int64_t)> const &fn) {
  if (ranges.empty()) {
    return;
  }
  if (ranges.size() == 1) {
    fn(ranges.front().first, ranges.front().second);
    return;
  }
  WorkerPool &pool = WorkerPool::instance();
  if (pool.worker_count() == 0) {
    for (auto const &range : ranges) {
      fn(range.first, range.second);
    }
    return;
  }
  Job job;
  job.ranges = &ranges;
  job.fn = &fn;
  pool.run(job);
}

}  // namespace orbitquant::cpu
