import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.managers.schedule_batch import release_req
from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestHiSparseRetractCpuCopy(unittest.TestCase):
    def test_cpu_copy_methods_accept_mamba_indices(self):
        for cls, method_name in (
            (HiSparseDSATokenToKVPool, "get_cpu_copy"),
            (HiSparseDSATokenToKVPool, "load_cpu_copy"),
            (HiSparseTokenToKVPoolAllocator, "get_cpu_copy"),
            (HiSparseTokenToKVPoolAllocator, "load_cpu_copy"),
        ):
            with self.subTest(cls=cls.__name__, method_name=method_name):
                signature = inspect.signature(getattr(cls, method_name))
                self.assertIn("mamba_indices", signature.parameters)

    def test_release_req_snapshots_hisparse_kv_before_retract(self):
        req = MagicMock()
        req.finished.return_value = False
        snapshot = {"kind": "hisparse", "host_len": 4, "host_kv": [], "index_k": None}
        call_order = []
        coordinator = MagicMock()
        coordinator.snapshot_kv_cache.side_effect = lambda _req: (
            call_order.append("snapshot") or snapshot
        )
        coordinator.retract_req.side_effect = lambda _req: call_order.append("retract")

        with (
            patch("sglang.srt.managers.schedule_batch.release_kv_cache"),
            patch("sglang.srt.managers.schedule_batch.evict_from_tree_cache"),
        ):
            release_req(
                req=req,
                remaing_req_count=1,
                server_args=SimpleNamespace(disaggregation_mode="decode"),
                req_to_token_pool=MagicMock(),
                token_to_kv_pool_allocator=MagicMock(),
                tree_cache=MagicMock(),
                hisparse_coordinator=coordinator,
            )

        self.assertEqual(call_order, ["snapshot", "retract"])
        self.assertEqual(req.kv_cache_cpu, snapshot)
        req.offload_kv_cache.assert_not_called()
        req.reset_for_retract.assert_called_once()

    def test_release_req_uses_device_offload_without_hisparse(self):
        req = MagicMock()
        req.finished.return_value = False
        req_to_token_pool = MagicMock()
        allocator = MagicMock()

        with (
            patch("sglang.srt.managers.schedule_batch.release_kv_cache"),
            patch("sglang.srt.managers.schedule_batch.evict_from_tree_cache"),
        ):
            release_req(
                req=req,
                remaing_req_count=1,
                server_args=SimpleNamespace(disaggregation_mode="decode"),
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                tree_cache=MagicMock(),
                hisparse_coordinator=None,
            )

        req.offload_kv_cache.assert_called_once_with(req_to_token_pool, allocator)

    def _make_resume_queue(self, req, coordinator, hisparse_available_size):
        queue = MagicMock()
        queue.retracted_queue = [req]
        queue._uses_swa_tail_prealloc.return_value = False
        queue._allocatable_token_budgets.return_value = 1000
        queue._prealloc_required_tokens.return_value = (1, 0)
        queue.req_to_token_pool.available_size.return_value = 1
        queue.scheduler.enable_hisparse = True
        queue.scheduler.hisparse_coordinator = coordinator
        queue.token_to_kv_pool_allocator.hisparse_attn_allocator.available_size.return_value = (
            hisparse_available_size
        )
        return queue

    def test_resume_retracted_reqs_readmits_hisparse_device_buffer(self):
        req = MagicMock()
        req.rid = "resume-readmit"
        req.kv_cache_cpu = {
            "kind": "hisparse",
            "host_len": 4,
            "host_kv": [],
            "index_k": None,
        }

        call_order = []
        coordinator = MagicMock()
        coordinator.padded_buffer_size = 8
        coordinator.load_kv_cache.side_effect = lambda _req: call_order.append("load")
        coordinator.admit_request_direct.side_effect = lambda _req: call_order.append(
            "admit"
        )

        queue = self._make_resume_queue(req, coordinator, hisparse_available_size=64)
        queue._pre_alloc.side_effect = lambda _req: call_order.append("prealloc")

        resumed_reqs = DecodePreallocQueue.resume_retracted_reqs(queue)

        self.assertEqual(resumed_reqs, [req])
        # Host KV must be restored before admission: admit_request_direct
        # preloads short sequences from the host pool.
        self.assertEqual(call_order, ["prealloc", "load", "admit"])
        self.assertFalse(hasattr(req, "kv_cache_cpu"))
        req.load_kv_cache.assert_not_called()

    def test_resume_retracted_reqs_respects_hisparse_device_budget(self):
        req = MagicMock()
        req.rid = "resume-budget"
        req.kv_cache_cpu = {
            "kind": "hisparse",
            "host_len": 4,
            "host_kv": [],
            "index_k": None,
        }

        coordinator = MagicMock()
        coordinator.padded_buffer_size = 8

        # Device pool has room for zero padded buffers: nothing may resume.
        queue = self._make_resume_queue(req, coordinator, hisparse_available_size=7)

        resumed_reqs = DecodePreallocQueue.resume_retracted_reqs(queue)

        self.assertEqual(resumed_reqs, [])
        self.assertEqual(queue.retracted_queue, [req])
        queue._pre_alloc.assert_not_called()
        coordinator.load_kv_cache.assert_not_called()
        coordinator.admit_request_direct.assert_not_called()


if __name__ == "__main__":
    unittest.main()
