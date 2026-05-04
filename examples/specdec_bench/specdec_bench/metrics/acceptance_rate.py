# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

from .base import Metric


class AcceptanceRate(Metric):
    def __init__(self, max_step_size=None):
        """Acceptance-rate / acceptance-length metric.

        Args:
            max_step_size: Optional cap on the number of tokens that a
                single speculative-decoding step can emit (typically
                ``draft_length + 1``). When set, any per-yield length
                ``L > max_step_size`` is split into
                ``K = ceil(L / max_step_size)`` synthetic steps with
                lengths ``[max_step_size] * (K-1) + [L - max_step_size *
                (K-1)]`` so total tokens are preserved. This corrects
                for async stream coalescing where the engine queues up
                multiple verify-step outputs and the consumer drains
                them in a single ``async for`` iteration -- without the
                split, those iterations are mis-counted as one giant
                step and inflate ``Average_AR`` and the
                ``Acceptance_Length_Histogram``. Leave as ``None`` for
                backward-compatible (uncorrected) behaviour.
        """
        super().__init__()
        self.prompt_ar = {}
        self.name = "acceptance_rate"
        self.max_step_size = max_step_size

    def _split_coalesced(self, length):
        """Yield per-step lengths for one coalesced async-stream yield."""
        m = self.max_step_size
        if m is None or length <= m:
            yield length
            return
        full, rem = divmod(length, m)
        for _ in range(full):
            yield m
        if rem:
            yield rem

    def process_step(self, step_outputs, request_id, turn_id):
        if request_id not in self.prompt_ar:
            self.prompt_ar[request_id] = {}
        if turn_id not in self.prompt_ar[request_id]:
            self.prompt_ar[request_id][turn_id] = []
        for i, beam_output in enumerate(step_outputs["output_ids"]):
            for output_id_iter in beam_output:
                for step_len in self._split_coalesced(len(output_id_iter)):
                    self.prompt_ar[request_id][turn_id].append(step_len)

    def _get_lengths(self, turn, lengths):
        for j in turn:
            if j not in lengths:
                lengths[j] = 0
            lengths[j] += 1

    def _process_lengths(self, lengths):
        lengths = dict(sorted(lengths.items(), key=lambda x: x[0]))
        self.out["Acceptance_Length_Histogram"] = lengths
        print("Acceptance Length Histogram")
        print(lengths)
        sum_lengths = sum(lengths.values())
        running_len = sum_lengths
        prev_ratio = 1
        self.out["Conditional_Acceptance_Rate"] = {}
        print("Conditional acceptance rate")
        for k, v in lengths.items():
            print(k, running_len / sum_lengths / prev_ratio)
            self.out["Conditional_Acceptance_Rate"][k] = running_len / sum_lengths / prev_ratio
            prev_ratio = running_len / sum_lengths
            running_len -= v

    def process_final(self, text_outputs):
        all_ar = []
        lengths = {}
        self.out["Request_AR"] = {}
        self.prompt_ar = dict(sorted(self.prompt_ar.items(), key=lambda x: x[0]))
        for request_id, turns in self.prompt_ar.items():
            self.out["Request_AR"][request_id] = {}
            for turn_id, turn in turns.items():
                ar = sum(turn) / len(turn)
                self.out["Request_AR"][request_id][turn_id] = ar
                all_ar.append(ar)
                self._get_lengths(turn, lengths)
                print(request_id, turn_id, self.out["Request_AR"][request_id][turn_id])
        average_ar = sum(all_ar) / len(all_ar)
        print("Average AR:", average_ar)
        self.out["Average_AR"] = average_ar
        self._process_lengths(lengths)
        self.write()
        self._format_write_output(text_outputs)

    def clear(self):
        self.prompt_ar = []

    def _format_write_output(self, outputs):
        with open(os.path.join(self.directory, "responses.jsonl"), "w") as outfile:
            for i, messages in enumerate(outputs):
                q_id = i
                out_line = {}
                out_line["question_id"] = q_id
                if messages[0]["role"] == "system":
                    out_line["system_prompt"] = messages[0]["content"]
                q_turns = [c["content"] for c in messages if c["role"] == "user"]
                a_turns = [c["content"] for c in messages if c["role"] == "assistant"]
                out_line["turns"] = q_turns
                out_line["choices"] = [{"index": 0, "turns": a_turns}]
                json.dump(out_line, outfile)
                outfile.write("\n")
