#!/usr/bin/env python3
import math

class NeuPIMsAllocator:
    def __init__(self, n_channels=32, tokens_per_tile=16):
        self.n_channels = n_channels
        self.tokens_per_tile = tokens_per_tile
        # Simple simulation: track how many rows are used per channel
        self.channel_depth = [0] * n_channels
        
    def allocate(self, required_tokens):
        """
        Strategy B applied to NeuPIMs:
        1. Power-of-2 allocation.
        2. Priority: Spread across Channels (Inter-Channel) -> Then Deepen Rows (Intra-Channel).
        """
        # 1. Determine Block Size (Strategy B: Power of 2)
        # Find smallest power of 2 >= required_tokens
        block_size = 16
        while block_size < required_tokens:
            block_size *= 2
            
        num_tiles_needed = max(1, block_size // self.tokens_per_tile)
        
        print(f"Request: {required_tokens} tokens -> Block Size: {block_size} tokens -> {num_tiles_needed} Tiles")
        
        allocation_map = {} # ch_id -> num_rows_added
        
        if num_tiles_needed <= self.n_channels:
            # Case 1: Fit within one row across multiple channels (Pure Inter-Channel)
            # Simple Round-Robin or Load Balancing for demo
            # We allocate 1 row in 'num_tiles_needed' channels
            
            # Find channels with minimum depth to balance load
            sorted_channels = sorted(range(self.n_channels), key=lambda i: self.channel_depth[i])
            selected_channels = sorted_channels[:num_tiles_needed]
            
            for ch in selected_channels:
                self.channel_depth[ch] += 1
                allocation_map[ch] = 1
                
            print(f"  -> Mode: Inter-Channel Spreading")
            print(f"  -> Allocated 1 row in channels: {selected_channels}")
            
        else:
            # Case 2: Need multiple rows per channel (Hybrid Inter + Intra)
            # First fill all 32 channels (Width), then increase Depth
            rows_per_channel = num_tiles_needed // self.n_channels
            remainder = num_tiles_needed % self.n_channels
            
            print(f"  -> Mode: Hybrid (Full Width + Depth Increase)")
            print(f"  -> Base depth increase: {rows_per_channel} rows per channel")
            
            for ch in range(self.n_channels):
                extra = 1 if ch < remainder else 0
                total_new_rows = rows_per_channel + extra
                self.channel_depth[ch] += total_new_rows
                allocation_map[ch] = total_new_rows
                
        return allocation_map

    def calculate_reduction_cost(self, allocation_map):
        """
        Estimate Reduction Overhead
        Assumptions:
        - d_k = 128 (FP16) -> 256 Bytes per vector
        - Intra-Channel: PIM Accumulation Latency (e.g. 10 cycles per row)
        - Inter-Channel: Bus Transfer (32 Bytes/cycle) + NPU Add
        """
        max_rows_in_batch = max(allocation_map.values())
        active_channels = len(allocation_map)
        
        # 1. Intra-Channel Reduction Cost
        # Each channel accumulates its 'max_rows_in_batch' partial results serially
        # Cost ~= (Rows - 1) * Accumulation_Latency
        intra_cost_cycles = (max_rows_in_batch - 1) * 10 
        
        # 2. Inter-Channel Reduction Cost
        # All active channels send 1 result vector to NPU
        # Vector Size = 128 * 2 Bytes = 256 Bytes
        # Bus Width = 32 Bytes/cycle (assumed from dram_req_size/freq)
        # Serialization: NPU receives from 32 channels. 
        # If simple bus: Serial transfer. If crossbar: Parallel transfer but NPU input limited.
        # Let's assume NPU can ingest 1 vector per cycle per port, or limited by bus.
        # Conservative: Serial transfer of 'active_channels' vectors
        transfer_cycles = active_channels * (256 / 32) 
        
        # NPU Accumulation: Add 'active_channels' vectors
        npu_compute_cycles = active_channels * 1 # Pipelined add
        
        inter_cost_cycles = transfer_cycles + npu_compute_cycles
        
        total_cost = intra_cost_cycles + inter_cost_cycles
        
        print(f"  -> Reduction Analysis:")
        print(f"     - Active Channels: {active_channels}")
        print(f"     - Max Rows/Channel: {max_rows_in_batch}")
        print(f"     - Intra-Channel Cost (PIM Accum): {intra_cost_cycles} cycles")
        print(f"     - Inter-Channel Cost (Transfer+NPU): {inter_cost_cycles} cycles")
        print(f"     - Total Reduction Overhead: {total_cost} cycles")

# Run Demo
allocator = NeuPIMsAllocator()

print("--- Scenario 1: Small Request (Fit in Channels) ---")
alloc = allocator.allocate(150) # Needs 150 -> Block 256 -> 16 Tiles
allocator.calculate_reduction_cost(alloc)
print()

print("--- Scenario 2: Large Request (Exceed Channels) ---")
alloc = allocator.allocate(800) # Needs 800 -> Block 1024 -> 64 Tiles -> 2 Rows/Channel
allocator.calculate_reduction_cost(alloc)
print()
