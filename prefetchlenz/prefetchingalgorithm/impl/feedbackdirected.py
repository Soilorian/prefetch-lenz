"""
Feedback Directed Prefetching by Srinath
"""

import logging
from typing import List

from prefetchlenz.prefetchingalgorithm.access.feedbackdirectedmemoryaccess import (
    FeedbackDirectedMemoryAccess,
)
from prefetchlenz.prefetchingalgorithm.memoryaccess import MemoryAccess
from prefetchlenz.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("prefetchLenz.FeedbackDirectedPrefetcher")


class FeedbackDirectedPrefetcher(PrefetchAlgorithm):
    def __init__(
        self,
        t_interval=8192,
        a_high=0.75,
        a_low=0.40,
        t_lateness=0.01,
        t_pollution=0.005,
    ):
        """
        Initializes the Feedback Directed Prefetcher.

        Args:
            t_interval (int): The number of "useful evictions" to define a sampling interval.
            a_high (float): The high accuracy threshold.
            a_low (float): The low accuracy threshold.
            t_lateness (float): The lateness threshold.
            t_pollution (float): The pollution threshold.
        """
        self.t_interval = t_interval
        self.t_interval_counter = 0

        # Hardware counters for the current interval
        self.pref_total_interval = 0
        self.used_total_interval = 0
        self.late_total_interval = 0
        self.demand_total_interval = 0
        self.pollution_total_interval = 0

        # Counters for the entire run (used for the rolling average update)
        self.pref_total = 0
        self.used_total = 0
        self.late_total = 0
        self.demand_total = 0
        self.pollution_total = 0

        # Prefetcher Configuration based on the paper's Table 1
        self.configs = {
            1: {"distance": 4, "degree": 1},
            2: {"distance": 8, "degree": 1},
            3: {"distance": 16, "degree": 2},
            4: {"distance": 32, "degree": 4},
            5: {"distance": 64, "degree": 4},
        }
        self.dyn_config_counter = 3  # Initial value is Middle-of-the-Road
        self.prefetch_distance = self.configs[self.dyn_config_counter]["distance"]
        self.prefetch_degree = self.configs[self.dyn_config_counter]["degree"]

        # Thresholds
        self.A_high = a_high
        self.A_low = a_low
        self.T_lateness = t_lateness
        self.T_pollution = t_pollution

        # State for a simple stride prefetcher and tracking
        self.last_address_per_pc = {}
        self.inflight_prefetches = {}  # {address: {'is_late': bool, 'is_useful': bool}}
        self.miss_addresses_in_cache = set()  # Simulates the pollution filter

    def init(self):
        """
        Initializes or resets all internal counters and state.
        """
        self.t_interval_counter = 0
        self.pref_total_interval = 0
        self.used_total_interval = 0
        self.late_total_interval = 0
        self.demand_total_interval = 0
        self.pollution_total_interval = 0

        self.pref_total = 0
        self.used_total = 0
        self.late_total = 0
        self.demand_total = 0
        self.pollution_total = 0

        self.dyn_config_counter = 3
        self.prefetch_distance = self.configs[self.dyn_config_counter]["distance"]
        self.prefetch_degree = self.configs[self.dyn_config_counter]["degree"]

        self.last_address_per_pc = {}
        self.inflight_prefetches = {}
        self.miss_addresses_in_cache = set()
        logger.info("FeedbackDirectedPrefetcher initialized.")

    def _update_counters(
        self,
        access: MemoryAccess,
        is_demand_access=True,
        is_prefetch_hit=False,
        prefetched_addresses=None,
    ):
        """
        A simplified method to update the counters based on a memory access and prefetch results.
        In a real system, this would be tied to the cache and MSHR.
        """
        if prefetched_addresses:
            for addr in prefetched_addresses:
                self.pref_total_interval += 1
                self.inflight_prefetches[addr] = {"is_late": False, "is_useful": False}

        if is_demand_access:
            # Check for late prefetches
            if (
                self.inflight_prefetches.get(access.address)
                and not self.inflight_prefetches[access.address]["is_useful"]
            ):
                # If a demand access hits an in-flight prefetch, it's a late prefetch hit
                self.late_total_interval += 1
                self.used_total_interval += 1
                self.inflight_prefetches[access.address]["is_useful"] = True

            # Check for pollution (simplified)
            if access.address in self.miss_addresses_in_cache:
                self.pollution_total_interval += 1
                self.miss_addresses_in_cache.remove(access.address)

            # This logic assumes a demand miss updates the total demand misses count
            # A demand access that misses in cache but is not a prefetch hit counts towards total demand misses
            if not is_prefetch_hit:
                self.demand_total_interval += 1
                self.miss_addresses_in_cache.add(
                    access.address
                )  # Simulates adding to the "pollution filter"

            # Check if this access was a useful prefetch
            if is_prefetch_hit:
                # If a prefetch hits, it's useful. Update the used-total counter.
                if (
                    self.inflight_prefetches.get(access.address)
                    and not self.inflight_prefetches[access.address]["is_useful"]
                ):
                    self.used_total_interval += 1
                    self.inflight_prefetches[access.address]["is_useful"] = True

    def _end_interval(self):
        """
        Computes metrics and adjusts prefetcher aggressiveness at the end of an interval.
        """
        # Update rolling averages (Equation 1 from paper)
        self.pref_total = (self.pref_total + self.pref_total_interval) / 2
        self.used_total = (self.used_total + self.used_total_interval) / 2
        self.late_total = (self.late_total + self.late_total_interval) / 2
        self.demand_total = (self.demand_total + self.demand_total_interval) / 2
        self.pollution_total = (
            self.pollution_total + self.pollution_total_interval
        ) / 2

        # Compute metrics
        accuracy = self.used_total / self.pref_total if self.pref_total > 0 else 0
        lateness = self.late_total / self.used_total if self.used_total > 0 else 0
        pollution = (
            self.pollution_total / self.demand_total if self.demand_total > 0 else 0
        )

        # Classify metrics based on thresholds
        acc_class = (
            "high"
            if accuracy >= self.A_high
            else ("low" if accuracy <= self.A_low else "medium")
        )
        late_class = "late" if lateness >= self.T_lateness else "not-late"
        poll_class = "polluting" if pollution >= self.T_pollution else "not-polluting"

        # Adjust aggressiveness based on the paper's Table 2
        new_config_counter = self.dyn_config_counter
        if acc_class == "high":
            if late_class == "late":
                new_config_counter = min(5, self.dyn_config_counter + 1)
            elif poll_class == "polluting":
                new_config_counter = max(1, self.dyn_config_counter - 1)
        elif acc_class == "medium":
            if poll_class == "polluting":
                new_config_counter = max(1, self.dyn_config_counter - 1)
            elif late_class == "late":
                new_config_counter = min(5, self.dyn_config_counter + 1)
        else:
            if poll_class == "polluting" or late_class == "late":
                new_config_counter = max(1, self.dyn_config_counter - 1)

        # Otherwise (not late, not polluting), no change is needed (Cases 3, 7, 11)

        if new_config_counter != self.dyn_config_counter:
            self.dyn_config_counter = new_config_counter
            self.prefetch_distance = self.configs[self.dyn_config_counter]["distance"]
            self.prefetch_degree = self.configs[self.dyn_config_counter]["degree"]
            logger.info(
                f"Aggressiveness changed to level {self.dyn_config_counter} (distance={self.prefetch_distance}, degree={self.prefetch_degree})"
            )

        # Reset interval counters
        self.pref_total_interval = 0
        self.used_total_interval = 0
        self.late_total_interval = 0
        self.demand_total_interval = 0
        self.pollution_total_interval = 0
        self.t_interval_counter = 0
        self.inflight_prefetches = {}

    def progress(
        self, access: FeedbackDirectedMemoryAccess, prefetch_hit: bool
    ) -> List[int]:
        """
        Processes a single memory access, updates metrics, and generates prefetch candidates.
        """
        # This is a simplification, as the paper's model requires knowing cache evictions and fills.
        # Here, we assume a "useful eviction" occurs after a certain number of accesses.
        # This is a heuristic to trigger the interval-based feedback mechanism.
        if self.t_interval_counter >= self.t_interval:
            self._end_interval()

        # Update counters for the current access
        # Assume a demand access if it's not a prefetch hit
        is_demand_access = access.demandMiss
        self._update_counters(
            access=access,
            is_demand_access=is_demand_access,
            is_prefetch_hit=prefetch_hit,
        )

        # Generate new prefetch candidates using a simple stride prefetcher as a base
        prefetches = []
        if access.pc in self.last_address_per_pc:
            last_addr = self.last_address_per_pc[access.pc]
            stride = access.address - last_addr
            if stride != 0:  # Only prefetch for a non-zero stride
                for i in range(1, self.prefetch_degree + 1):
                    prefetch_addr = access.address + (i * stride)
                    prefetches.append(prefetch_addr)

        self.last_address_per_pc[access.pc] = access.address

        # Simulate sending the new prefetches and updating counters
        self._update_counters(access=access, prefetched_addresses=prefetches)

        # Update the interval counter. Assuming one useful eviction per memory access for simplicity
        self.t_interval_counter += 1

        return prefetches

    def close(self):
        """
        Cleans up any state at the end of the simulation.
        """
        self._end_interval()  # Final update at the end of the simulation
        logger.info("FeedbackDirectedPrefetcher closed.")
