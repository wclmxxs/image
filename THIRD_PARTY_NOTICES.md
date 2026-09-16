# Third-party notices

`image_lab/ideogram_fast.py` adapts the forward and sampling equations from
Hugging Face Diffusers v0.39.0, `transformer_ideogram4.py` and
`pipeline_ideogram4.py`.

Copyright 2026 Ideogram AI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not
use this material except in compliance with the License. You may obtain a copy at
https://www.apache.org/licenses/LICENSE-2.0 (also included in
`licenses/Apache-2.0.txt`). Unless required by applicable law
or agreed to in writing, software distributed under the License is distributed
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for the specific language governing permissions and
limitations under the License.

Changes: specialize to single-image, eight-step Instant inference; validate and
remove isolated left padding; precompute fixed conditioning; omit the distilled
zero unconditional branch; optionally compile repeated transformer blocks.
Model weights retain their own licenses.
