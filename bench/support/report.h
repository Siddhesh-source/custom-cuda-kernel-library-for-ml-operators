#ifndef CCML_BENCH_SUPPORT_REPORT_H
#define CCML_BENCH_SUPPORT_REPORT_H

#include <string>

#include "bench/support/timer.h"

namespace ccml_bench {

/*
 * CSV output, so results can be diffed between runs and between machines.
 *
 * Deliberately emits elapsed time only. Derived figures -- achieved throughput,
 * bandwidth utilization, ratios against any other library -- are analysis, and
 * belong in docs/benchmarking.md next to the conditions they were measured
 * under. Putting them here would make it far too easy for a number with no
 * provenance to end up in a report.
 */

struct Record {
  std::string  op;
  std::string  algo;
  std::string  shape;   /* e.g. "M=128,N=128,K=128" */
  TimingResult timing;
};

/* Prints the run's conditions: device, library version, build type, date.
   Every CSV must be traceable to the machine that produced it. */
void print_run_header();

void print_csv_header();
void print_csv_row(const Record& record);

}  // namespace ccml_bench

#endif /* CCML_BENCH_SUPPORT_REPORT_H */
