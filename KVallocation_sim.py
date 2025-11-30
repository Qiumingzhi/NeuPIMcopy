#!/usr/bin/env python3
"""示例说明为什么细粒度策略分配次数更少"""

def find_largest_block_le(required: int, size_classes: list[int]) -> int:
    """找到小于等于需求的最大块"""
    best_block = 0
    for size in size_classes:
        if size <= required:
            best_block = size
        else:
            break
    if best_block > 0:
        return best_block
    return size_classes[0]

def allocate_blocks(required_tokens: int, size_classes: list[int]) -> list[int]:
    """模拟分配过程"""
    blocks = []
    allocated = 0
    while allocated < required_tokens:
        remaining = required_tokens - allocated
        block = find_largest_block_le(remaining, size_classes)
        blocks.append(block)
        allocated += block
    return blocks

# 策略B的大小类
strategy_b_classes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

# 策略C的大小类（细粒度）
strategy_c_classes = list(range(16, 256 + 1, 16)) + list(range(320, 4096 + 1, 64))

print("策略B大小类:", strategy_b_classes[:10], "...")
print("策略C大小类:", strategy_c_classes[:20], "...")
print()

# 测试几个不同的需求大小
test_cases = [150, 200, 300, 500, 800, 1200]

print("=" * 70)
print(f"{'需求tokens':<12} {'策略B':<30} {'策略C':<30}")
print(f"{'':<12} {'块数':<6} {'块列表':<24} {'块数':<6} {'块列表':<24}")
print("=" * 70)

for required in test_cases:
    blocks_b = allocate_blocks(required, strategy_b_classes)
    blocks_c = allocate_blocks(required, strategy_c_classes)
    
    blocks_b_str = "+".join(map(str, blocks_b))
    blocks_c_str = "+".join(map(str, blocks_c))
    
    print(f"{required:<12} {len(blocks_b):<6} {blocks_b_str:<24} {len(blocks_c):<6} {blocks_c_str:<24}")

print()
print("关键观察：")
print("1. 策略C有更多中间大小（如144, 160, 176等），能更精确匹配需求")
print("2. 策略B只有指数增长的大小，经常需要多个小块来填补")
print("3. 例如：需要150 tokens时")
print("   - 策略B: 128 + 16 + 16 = 3块")
print("   - 策略C: 144 + 16 = 2块（更少！）")
