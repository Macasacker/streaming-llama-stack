"""Linux paging counters: /proc readers, their units, and the guard's receipt shape."""
import mmap
import unittest

from slstack.memory_counters import (available_physical_bytes, diagnostic_snapshot, meminfo_bytes,
                                     pagefile_used_bytes, snapshot, swap_bytes, vmstat_counters)

MEMINFO = """MemTotal:       16384000 kB
MemFree:         2048000 kB
MemAvailable:    9000000 kB
SwapTotal:       4194304 kB
SwapFree:        3145728 kB
SwapCached:            0 kB
"""

VMSTAT = """pgpgin 1000
pgpgout 500
pswpin 2
pswpout 4
pgmajfault 7
"""


class ProcfsReaderTests(unittest.TestCase):
    def test_meminfo_fields_are_bytes(self):
        info = meminfo_bytes(MEMINFO)
        self.assertEqual(info['SwapTotal'], 4 * 1024**3)
        self.assertEqual(info['MemAvailable'], 9000000 * 1024)

    def test_swap_use_is_the_difference_of_the_two_totals(self):
        self.assertEqual(swap_bytes(meminfo_bytes(MEMINFO)), (1024**3, 4 * 1024**3))

    def test_a_box_without_swap_reports_zero(self):
        info = meminfo_bytes(MEMINFO.replace('SwapFree:        3145728 kB', 'SwapFree:              0 kB')
                             .replace('SwapTotal:       4194304 kB', 'SwapTotal:             0 kB'))
        self.assertEqual(swap_bytes(info), (0, 0))

    def test_vmstat_pgpg_counters_are_512_byte_blocks(self):
        counters = vmstat_counters(VMSTAT)
        self.assertEqual(counters['pages_input']['bytes'], 1000 * 512)
        self.assertEqual(counters['pages_output']['bytes'], 500 * 512)

    def test_swap_counters_are_pages(self):
        counters = vmstat_counters(VMSTAT)
        self.assertEqual(counters['page_reads']['bytes'], 2 * mmap.PAGESIZE)
        self.assertEqual(counters['page_writes']['bytes'], 4 * mmap.PAGESIZE)

    def test_major_faults_are_a_count_not_a_byte_total(self):
        counters = vmstat_counters(VMSTAT)
        self.assertEqual(counters['major_faults'], dict(count=7, raw=7))

    def test_a_missing_field_is_reported_not_invented_as_zero(self):
        counters = vmstat_counters('pgpgin 10\n')
        self.assertEqual(counters['pages_input']['bytes'], 5120)
        self.assertIn('vmstat field missing', counters['pages_output']['error'])
        self.assertNotIn('bytes', counters['pages_output'])


class LiveCounterTests(unittest.TestCase):
    def test_available_physical_ram_is_positive(self):
        self.assertGreater(available_physical_bytes(), 0)

    def test_live_swap_use_never_exceeds_its_total(self):
        used, total = swap_bytes()
        self.assertLessEqual(used, total)
        self.assertEqual(pagefile_used_bytes(), used)

    def test_guard_snapshot_carries_the_receipt_fields(self):
        row = snapshot()
        self.assertEqual(row['api'], '/proc/meminfo')
        self.assertEqual(row['paging_file_usage'], dict(first=row['paging_file_usage']['first'],
                                                        second=row['paging_file_usage']['second']))
        self.assertGreaterEqual(row['pagefile_count'], 0)
        self.assertGreater(row['available_physical_bytes'], 0)

    def test_diagnostic_snapshot_reports_every_counter_the_check_gate_reads(self):
        row = diagnostic_snapshot()
        for name in ('pages_input', 'pages_output', 'page_reads', 'page_writes'):
            self.assertIn('bytes', row[name], name)
        self.assertIn('paging_file_usage', row)


if __name__ == '__main__':
    unittest.main()
