#!/usr/bin/env python3
"""
Simple OHLC Constraint Fix Test
"""

import numpy as np

def fix_ohlc_simple(ohlc_data):
    """Simple OHLC constraint fixing"""
    fixed = np.copy(ohlc_data)
    
    for i in range(len(ohlc_data)):
        values = ohlc_data[i].copy()
        sorted_vals = np.sort(values)
        
        # Assign: Low=min, High=max, Open=2nd lowest, Close=2nd highest
        fixed[i] = [sorted_vals[1], sorted_vals[3], sorted_vals[0], sorted_vals[2]]
        
        o, h, l, c = fixed[i]
        print(f"Sample {i}: O={o:.3f}, H={h:.3f}, L={l:.3f}, C={c:.3f}")
        
        # Verify constraints
        assert h >= max(o, c), f"High constraint violated"
        assert l <= min(o, c), f"Low constraint violated"
    
    return fixed

# Test
np.random.seed(42)
test_data = np.random.normal(100, 5, (3, 4))

print("Original OHLC:")
for i, (o, h, l, c) in enumerate(test_data):
    print(f"Sample {i}: O={o:.3f}, H={h:.3f}, L={l:.3f}, C={c:.3f}")
    if h < max(o, c) or l > min(o, c):
        print(f"  ❌ Constraint violation!")

print("\nFixed OHLC:")
fixed_data = fix_ohlc_simple(test_data)

print("\n✅ All constraints satisfied!")
