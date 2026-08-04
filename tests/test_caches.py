from typing import override
import unittest
import numpy as np
from evantrace.caches import Cache
from evantrace.caches import MAIN_MEMORY_LATENCY
from evantrace.replacement_policies import ReplacementPolicy

class Test4WayLRUCache(unittest.TestCase):
    """
    Tests the Cache class.
    """
    
    # Class-level type hint
    cache: 'Cache'

    def __init__(self, *args, **kwargs):
        """Initialize with a default cache to satisfy linters."""
        super().__init__(*args, **kwargs)
        # Use default values for the linter's placeholder
        self.cache = Cache()
        self.read_latency: int = 0
        self.write_latency: int = 0
        self.test_addr: np.uint64 = np.uint64(0xcccccccccccccccc)
        self.test_addr_miss: np.uint64 = np.uint64(0xcccccccccccccccc)

    @override
    def setUp(self):
        """
        Set up a specific cache geometry for testing.
        This is a 16KB, 4-way set associative cache.
        - 16384 bytes / 64 bytes/line = 256 lines
        - 256 lines / 4 lines/set = 64 sets
        - index_bits = log2(64) = 6
        - offset_bits = log2(64) = 6
        - tag_bits = 64 - 6 - 6 = 52
        """
        self.read_latency = 4
        self.write_latency = 0
        self.cache = Cache(
            associativity=4,
            line_size=64,
            total_size=16384, # Use a power of 2 for easy testing
            replacement_policy=ReplacementPolicy.LRU,
            read_latency=self.read_latency,
            write_latency=self.write_latency,
            parent=None
        )
        
        # --- Pre-load a line into the cache to test hits ---
        
        # We'll pick an address
        self.test_addr_hit = np.uint64(0x12345000DECAFBAD)
        self.test_addr_miss = np.uint64(0xFEEDFACECAFEBABE)
        
        # Manually calculate its parts based on our geometry
        # offset = 6 bits, index = 6 bits
        # Tag: 0x12345000DECAF (52 bits)
        # Index: 0x2A (binary 101010) (6 bits)
        # Offset: 0x2D (binary 101101) (6 bits)
        
        # Get the parts programmatically
        test_tag, test_index, _ = self.cache.get_address_parts(self.test_addr)
        
        # Manually set this line to be valid and present
        # We'll put it in line 0 of its set
        self.cache.valid[test_index][0] = True
        self.cache.tags[test_index][0] = test_tag
        
        # Manually update metadata to make it MRU (age 0)
        # [3, 2, 1, 0] -> [3, 2, 1, 0] (no change)
        # Let's just set it to 0
        self.cache.metadata[test_index] = np.array([3, 2, 1, 0], dtype=np.uint32)
        
        # --- Define an address for misses ---
        self.test_addr_miss = np.uint64(0xFEEDFACECAFEBABE)
        # Ensure this address is not the same as the hit address
        # by checking tag and index
        self.assertNotEqual(
            self.cache.get_address_parts(self.test_addr_hit),
            self.cache.get_address_parts(self.test_addr_miss),
            "Test hit and miss addresses map to the same set and tag!"
        )

        # --- Define addresses for set fill and eviction test ---
        # We need 5 addresses that map to the *same index* but *different tags*.
        # Our index bits = 6. Let's pick index 0x1A (26).
        # Offset bits = 6.
        # Addr = (Tag << 12) | (Index << 6) | (Offset)
        
        self.fill_addrs = []
        base_addr = (0x1A << 6) # Index 0x1A, Offset 0
        for i in range(1, 5): # Tags 1, 2, 3, 4
            self.fill_addrs.append( (i << 12) | base_addr )
        
        # This will be the 5th address to trigger eviction
        self.evict_addr = np.uint64((5 << 12) | base_addr) # Tag 5
        
        # Verify all addresses map to the same index
        idx_set = set(self.cache.get_address_parts(addr)[1] for addr in self.fill_addrs)
        idx_set.add(self.cache.get_address_parts(self.evict_addr)[1])
        self.assertEqual(len(idx_set), 1, "Fill/Evict addresses do not map to the same set")


    def test_simple_read_hit_latency(self):
        """
        Tests that a read to a pre-loaded address is a hit
        and returns only the cache's read_latency.
        """
        # self.test_addr was loaded in setUp
        expected_latency = self.read_latency
        
        latency = self.cache.read(self.test_addr)
        
        self.assertEqual(latency, expected_latency)

    def test_simple_write_hit_latency(self):
        """
        Tests that a write to a pre-loaded address is a hit
        and returns only the cache's write_latency.
        """
        # self.test_addr was loaded in setUp
        expected_latency = self.write_latency
        
        latency = self.cache.write(self.test_addr)
        
        self.assertEqual(latency, expected_latency)
        
    def test_read_miss_latency(self):
            """
            Tests that a read to an unloaded address is a miss
            and returns L1 read_latency + Main Memory latency.
            """
            # self.test_addr_miss is not in the cache
            expected_latency = self.read_latency + MAIN_MEMORY_LATENCY
            
            latency = self.cache.read(self.test_addr_miss)
            
            self.assertEqual(latency, expected_latency)
            
            # test whether cache state was updated to add test_addr_miss
            expected_latency = self.read_latency
            latency = self.cache.read(self.test_addr_miss)
            
            self.assertEqual(latency, expected_latency)
    
    def test_write_miss_latency(self):
        """
        Tests that a write to an unloaded address is a miss (Write-Allocate)
        and returns L1 write_latency + Main Memory latency.
        """
        # self.test_addr_miss is not in the cache
        # Latency is write_latency (for the hit) + read_latency (for the fetch)
        expected_latency = self.write_latency
        
        latency = self.cache.write(self.test_addr_miss)
        
        self.assertEqual(latency, expected_latency)
        
        # write latency should be the same whether address is a hit or a miss
        latency = self.cache.write(self.test_addr_miss)
        
        self.assertEqual(latency, expected_latency)
        
        # reading from the now written address should be a hit
        expected_latency = self.read_latency
        latency = self.cache.read(self.test_addr_miss)
        
        self.assertEqual(latency, expected_latency)
        
    def test_set_fill_and_eviction(self):
            """
            Tests filling a set to capacity and then evicting a line.
            Uses the LRU replacement policy.
            """
            miss_latency = self.read_latency + MAIN_MEMORY_LATENCY
            hit_latency = self.read_latency
            
            # --- 1. Fill the set (4 lines) ---
            # These should all be misses
            for i, addr in enumerate(self.fill_addrs):
                lat = self.cache.read(addr)
                self.assertEqual(lat, miss_latency, f"Fill address {i} was not a miss")
            
            # After this, the LRU state for the set should be:
            # fill_addrs[0] = age 3 (LRU)
            # fill_addrs[1] = age 2
            # fill_addrs[2] = age 1
            # fill_addrs[3] = age 0 (MRU)
            
            # --- 2. Check for hits ---
            # Reading them all again should all be hits
            for i, addr in enumerate(self.fill_addrs):
                lat = self.cache.read(addr)
                self.assertEqual(lat, hit_latency, f"Hit check address {i} was not a hit")
                
            # After this, the LRU state for the set should be:
            # fill_addrs[0] = age 3 (LRU)
            # fill_addrs[1] = age 2
            # fill_addrs[2] = age 1
            # fill_addrs[3] = age 0 (MRU)
            # (This is because we read them in the same order)
    
            # --- 3. Evict an entry ---
            # Accessing a 5th address in the same set should be a miss
            # and should evict the LRU line (fill_addrs[0])
            lat = self.cache.read(self.evict_addr)
            self.assertEqual(lat, miss_latency, "Eviction access was not a miss")
            
            # New LRU state:
            # fill_addrs[1] = age 3 (LRU)
            # fill_addrs[2] = age 2
            # fill_addrs[3] = age 1
            # evict_addr  = age 0 (MRU)
            
            # --- 4. Check that the victim is gone ---
            # Accessing the original LRU line (fill_addrs[0]) should now be a miss
            lat = self.cache.read(self.fill_addrs[0])
            self.assertEqual(lat, miss_latency, "Victim was not evicted (LRU error)")
            
            # --- 5. Check that non-victims are still present ---
            # Accessing any other line (e.g., fill_addrs[3]) should still be a hit
            lat = self.cache.read(self.fill_addrs[3])
            self.assertEqual(lat, hit_latency, "A non-victim line was evicted")


def run_tests():
    """Run all tests and print results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(Test4WayLRUCache))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\n{'='*70}")
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"{'='*70}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
