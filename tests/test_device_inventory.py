"""Pure-Python inventory checks: no CUDA imports, kernel compilation or solves."""
import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    'device_inventory_under_test', Path(__file__).parents[1] / 'hdgfem/backends/device_inventory.py')
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


class FakeRuntime:
    def __init__(self):
        self.current = 1

    def getDeviceCount(self):
        return 3

    def getDevice(self):
        return self.current

    def setDevice(self, device):
        self.current = device

    def getDeviceProperties(self, device):
        major, minor, sms = ((3, 7, 13), (6, 0, 56), (7, 0, 80))[device]
        return dict(name=(b'K80 half', b'P100', b'V100')[device], major=major,
                    minor=minor, multiProcessorCount=sms, totalGlobalMem=16 * 2**30)

    def deviceGetAttribute(self, attribute, device):
        return (875000, 1328000, 1380000)[device] if attribute == 13 else (3, 2, 2)[device]

    def memGetInfo(self):
        return 14 * 2**30, 16 * 2**30


class DeviceInventoryTests(unittest.TestCase):
    def test_mesocentre_ranking_and_context_restore(self):
        runtime = FakeRuntime()
        records = inventory.discover_cuda_devices(runtime)
        self.assertEqual(runtime.current, 1)
        self.assertEqual(inventory.select_fp64_device(records)['name'], 'V100')
        self.assertAlmostEqual(records[2]['fp64_peak_flops'], 7.0656e12)
        self.assertEqual(len(records), 3)

    def test_memory_is_per_device(self):
        records = inventory.discover_cuda_devices(FakeRuntime())
        with self.assertRaisesRegex(ValueError, 'No usable'):
            inventory.select_fp64_device(records, minimum_free_bytes=20 * 2**30)

    def test_unknown_architecture_requires_override(self):
        self.assertIsNone(inventory.fp64_peak_flops(99, 0, 100, 1000000, 2))
        records = [dict(device=0, usable=True, free_bytes=100, fp64_peak_flops=None)]
        with self.assertRaisesRegex(ValueError, 'Unknown FP64'):
            inventory.select_fp64_device(records)
        self.assertEqual(inventory.select_fp64_device(records, overrides={0: 1e12})['device'], 0)

    def test_override_cannot_escape_scheduler_mask(self):
        records = inventory.discover_cuda_devices(FakeRuntime())
        with self.assertRaisesRegex(ValueError, 'visibility'):
            inventory.select_fp64_device(records, overrides={4: 1e15})

    def test_unavailable_device_does_not_mask_others(self):
        class BrokenRuntime(FakeRuntime):
            def getDeviceProperties(self, device):
                if device == 2:
                    raise RuntimeError('device unavailable')
                return super().getDeviceProperties(device)
        runtime = BrokenRuntime()
        rows = inventory.discover_cuda_devices(runtime)
        self.assertFalse(rows[2]['usable'])
        self.assertEqual(inventory.select_fp64_device(rows)['device'], 1)
        self.assertEqual(runtime.current, 1)


if __name__ == '__main__':
    unittest.main()
