#ifndef MODELS_H
#define MODELS_H

#include <cstdint>
#include <map>
#include <vector>

#include "instr.h"

namespace analytical {

// resp_cycle from Algorithm 1
uint64_t resp_cycle(uint64_t req_cycle, Instr instr,
                    std::map<uint64_t, uint64_t> last_req_cycles,
                    std::map<uint64_t, uint64_t> last_resp_cycles);

double get_thr_rob(const std::vector<Instr>& window, uint16_t rob_size);
double get_thr_load_queue(const std::vector<Instr>& window,
                          uint16_t load_queue_size);
double get_thr_store_queue(const std::vector<Instr>& window,
                           uint16_t store_queue_size);
double get_thr_alu_issue(const std::vector<Instr>& window,
                         uint16_t alu_issue_width);
double get_thr_alu_mult_div_issue(const std::vector<Instr>& window,
                                  uint16_t alu_mult_div_issue_width);
double get_thr_fp_issue(const std::vector<Instr>& window,
                        uint16_t fp_issue_width);
double get_thr_fp_mult_div_issue(const std::vector<Instr>& window,
                                 uint16_t fp_mult_div_issue_width);
double get_thr_ls_issue(const std::vector<Instr>& window,
                        uint16_t ls_issue_width);
double get_thr_load_ls_pipes_lower(const std::vector<Instr>& window,
                                   uint16_t num_ls_pipes,
                                   uint16_t num_load_pipes);
double get_thr_load_ls_pipes_upper(const std::vector<Instr>& window,
                                   uint16_t num_ls_pipes,
                                   uint16_t num_load_pipes);
double get_thr_icache_fills(const std::vector<Instr>& window,
                            uint16_t max_icache_fills);

}  // namespace analytical

#endif  // MODELS_H
