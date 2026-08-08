"""Unit smoke tests for the minimal DSpark PD metadata-buffer PoC."""

import unittest

import torch

from sglang.srt.disaggregation.utils import MetadataBuffers
from sglang.srt.speculative.dspark_components.dspark_disaggregation import (
    extract_dspark_pd_hidden_from_logits,
    resolve_dspark_pd_prefill_tail_len,
)


class TestDSparkPdMetadataPoc(unittest.TestCase):
    def test_extract_tail_hidden(self):
        hidden = torch.arange(12, dtype=torch.float32).view(4, 3)
        last, tail, mask = extract_dspark_pd_hidden_from_logits(
            hidden_states=hidden,
            token_start=0,
            token_end=4,
            tail_len=3,
        )
        self.assertEqual(tuple(last.shape), (3,))
        self.assertTrue(torch.equal(last, hidden[-1].cpu()))
        self.assertEqual(tuple(tail.shape), (3, 3))
        self.assertTrue(torch.equal(tail[-3:], hidden[-3:].cpu()))
        self.assertTrue(bool(mask.all()))

    def test_metadata_buffers_roundtrip_tail(self):
        bufs = MetadataBuffers(
            size=2,
            hidden_size=4,
            hidden_states_dtype=torch.float16,
            dspark_prefill_tail_len=3,
        )

        class _Req:
            metadata_buffer_index = 0
            output_ids = [7]
            cached_tokens = 1
            cached_tokens_device = 1
            cached_tokens_host = 0
            cached_tokens_storage = 0
            multimodal_inputs = None
            return_logprob = False
            return_sampling_mask = False
            bootstrap_room = 123
            hidden_states_tensor = torch.ones(4, dtype=torch.float16)
            output_topk_p = torch.zeros(1, dtype=torch.float32)
            output_topk_index = torch.zeros(1, dtype=torch.int64)
            output_dsa_topk_indices = None
            prefill_tail_hidden_states_tensor = torch.arange(
                12, dtype=torch.float16
            ).view(3, 4)
            prefill_tail_valid_mask = torch.tensor([True, True, True])

        bufs.set_buf(_Req())
        packed = bufs.get_buf(0)
        # Indices follow MetadataBuffers.get_buf layout.
        self.assertTrue(torch.equal(packed[11], torch.ones(4, dtype=torch.float16)))
        self.assertEqual(tuple(packed[12].shape), (3, 4))
        self.assertTrue(bool(packed[13].all()))
        self.assertEqual(int(packed[-1][0].item()), 123)

    def test_default_tail_len(self):
        self.assertGreaterEqual(resolve_dspark_pd_prefill_tail_len(), 1)


if __name__ == "__main__":
    unittest.main()
