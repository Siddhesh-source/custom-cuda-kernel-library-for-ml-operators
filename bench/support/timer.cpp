#include "bench/support/timer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "bench/support/spin.h"
#include "support/cuda_event.h"

namespace ccml_bench {

namespace {

void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

/* Every launch is followed by cudaGetLastError (reference.md, Method). */
bool launch_checked(const Candidate& c) {
  if (!c.launch()) {
    return false;
  }
  check(cudaGetLastError(), "kernel launch");
  return true;
}

}  // namespace

L2Flusher::L2Flusher(std::size_t l2_bytes) : bytes_(2 * l2_bytes) {
  if (bytes_ == 0) {
    bytes_ = 8u << 20;
  }
  check(cudaMalloc(&buffer_, bytes_), "cudaMalloc(L2 flush buffer)");
}

L2Flusher::~L2Flusher() { cudaFree(buffer_); }

void L2Flusher::flush(cudaStream_t stream) {
  check(cudaMemsetAsync(buffer_, 0x5a, bytes_, stream), "cudaMemsetAsync(L2 flush)");
}

int choose_iterations(double estimate_ms, const TimingConfig& config) {
  if (!(estimate_ms > 0.0)) {
    return config.max_iters;
  }
  const double fit = config.budget_ms / estimate_ms;
  const int n = fit >= config.max_iters ? config.max_iters : static_cast<int>(fit);
  return std::max(config.min_iters, std::min(config.max_iters, n));
}

std::vector<Measured> time_interleaved(const std::vector<Candidate>& candidates,
                                       const TimingConfig& config,
                                       cudaStream_t stream,
                                       L2Flusher* flusher,
                                       const TelemetrySource* telemetry) {
  std::vector<Measured> out(candidates.size());
  std::vector<std::vector<double>> samples(candidates.size());
  std::vector<int> iters(candidates.size(), config.max_iters);
  std::vector<bool> alive(candidates.size(), true);
  const bool cold = config.cold_l2 && flusher != nullptr;

  auto fail = [&](std::size_t i, const std::string& why) {
    alive[i] = false;
    out[i].valid = false;
    out[i].error = why;
  };

  /* Warmup, and a per-call estimate from the batched pattern of the spec. */
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    try {
      for (int w = 0; w < config.warmup; ++w) {
        if (!launch_checked(candidates[i])) {
          fail(i, "not run");
          break;
        }
      }
      if (!alive[i]) continue;
      check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

      ccml_support::CudaEvent a, b;
      const int probe = 5;
      a.record(stream);
      for (int k = 0; k < probe; ++k) launch_checked(candidates[i]);
      b.record(stream);
      b.synchronize();
      iters[i] = choose_iterations(b.elapsed_ms_since(a) / probe, config);
    } catch (const std::exception& e) {
      fail(i, e.what());
    }
  }

  /* Interleaved, per-iteration timed rounds. */
  const int rounds = std::max(1, config.rounds);
  for (int r = 0; r < rounds; ++r) {
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      if (!alive[i]) continue;
      const int per_round = (iters[i] + rounds - 1) / rounds;
      try {
        std::vector<ccml_support::CudaEvent> start(static_cast<std::size_t>(per_round));
        std::vector<ccml_support::CudaEvent> stop(static_cast<std::size_t>(per_round));
        for (int k = 0; k < per_round; ++k) {
          if (cold) flusher->flush(stream);
          launch_spin(stream, config.guard_us);
          start[static_cast<std::size_t>(k)].record(stream);
          if (!launch_checked(candidates[i])) {
            throw std::runtime_error("launch stopped succeeding mid-run");
          }
          stop[static_cast<std::size_t>(k)].record(stream);
        }
        /* Sample while the round is still executing (the guard alone keeps
           the GPU busy for per_round * guard_us), on the last round. */
        if (telemetry != nullptr && r == rounds - 1) {
          out[i].telemetry = telemetry->sample();
        }
        stop.back().synchronize();
        for (int k = 0; k < per_round; ++k) {
          const auto idx = static_cast<std::size_t>(k);
          samples[i].push_back(1e3 * stop[idx].elapsed_ms_since(start[idx]));
        }
      } catch (const std::exception& e) {
        fail(i, e.what());
      }
    }
  }

  /* The spec's batched event pattern, warm cache only (a flush between
     launches would be inside the timed region). */
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (!alive[i]) continue;
    try {
      double batched = 0.0;
      if (!cold) {
        ccml_support::CudaEvent a, b;
        a.record(stream);
        for (int k = 0; k < iters[i]; ++k) launch_checked(candidates[i]);
        b.record(stream);
        b.synchronize();
        batched = 1e3 * b.elapsed_ms_since(a) / iters[i];
      }
      out[i].summary = summarize(samples[i]);
      out[i].batched_mean_us = batched;
      out[i].valid = out[i].summary.valid && out[i].summary.n >= config.min_iters;
      if (!out[i].valid && out[i].error.empty()) {
        out[i].error = "fewer than min_iters samples";
      }
    } catch (const std::exception& e) {
      fail(i, e.what());
    }
  }
  return out;
}

}  // namespace ccml_bench
