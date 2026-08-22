# Third-Party Licences

This project reuses components from several external projects. Their licences are
**not uniform**: two are MIT, one is Apache-2.0 or 3-clause BSD depending on
version, one is AGPL-3.0, one set of model weights carries no stated licence, and
one dataset is governed by terms separate from the MIT licence of the repository
that distributes it. Each is recorded below, with what was reused and what the
licence requires. Licence texts are reproduced in full where the licence requires
it and where length permits; the AGPL-3.0 is linked rather than inlined, for the
reason given in that section.

This project's own original code is released under the MIT Licence
(`LICENSE`, Copyright (c) 2026 Sreeshanth Sivanantham).

## Summary

| Component | What is reused | Licence |
|---|---|---|
| chandrikadeb7/Face-Mask-Detection | MobileNetV2 mask classifier, dataset, procedure | MIT |
| OpenCV res10 face detector | `deploy.prototxt` (redistributed here) | Apache-2.0 (OpenCV ≥ 4.5.0); 3-clause BSD (≤ 4.4.0) |
| OpenCV res10 face detector | `res10_300x300_ssd_iter_140000.caffemodel` | **No licence stated upstream** |
| saimj7/Social-Distancing-Detection-in-Real-Time | YOLOv3 config, weights, Darknet approach | MIT |
| Ultralytics (`ultralytics>=8.1`) | YOLOv8n detector and ByteTrack tracker | **AGPL-3.0** (or paid commercial licence) |
| BAFMD — repository code | Annotation format and download tooling | MIT |
| BAFMD — dataset images | Test images and Pascal VOC annotations | **Non-commercial research only**, plus platform terms |

Two entries are highlighted because they are the ones that constrain reuse of this
project rather than merely requiring attribution.

---

## chandrikadeb7/Face-Mask-Detection

https://github.com/chandrikadeb7/Face-Mask-Detection

**Reused:** the trained `mask_detector.model` MobileNetV2 classifier; the
res10 SSD face-detector assets (`deploy.prototxt`, `res10_300x300_ssd_iter_140000.caffemodel`);
the 4,092-image mask dataset; and the face-detection/mask-classification
procedure adapted in `src/mask_classifier.py`.

**Note:** the face-detector assets originate from OpenCV's `samples/dnn/face_detector`
and were obtained via this project. `models/deploy.prototxt` is byte-identical
to the copy distributed there. Their licence position is therefore not settled by
the MIT licence below; see the following section.

```
MIT License

Copyright (c) 2021 chandrikadeb7

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## OpenCV res10 SSD face detector

https://github.com/opencv/opencv/tree/4.x/samples/dnn/face_detector

**Reused:** `models/deploy.prototxt`, redistributed in this repository and
byte-identical to the copy in the OpenCV repository, and the accompanying
`res10_300x300_ssd_iter_140000.caffemodel` weights.

**Licence of the configuration file.** OpenCV versions 4.5.0 and higher are
licensed under the Apache 2.0 Licence; versions 4.4.0 and lower, including the
3.x and 2.x series, are licensed under the 3-clause BSD Licence. The prototxt is
part of the OpenCV repository and is covered accordingly.

**Licence of the trained weights — unresolved upstream.** The `.caffemodel`
weights are not stored in the OpenCV repository; they are retrieved by a download
script from a third-party location and carry no licence statement of their own.
The absence of a licence for these models has been raised on OpenCV's own
community forum and, as far as could be established, remains unanswered. This
project therefore records the position rather than asserting one: the weights are
used here for academic evaluation, they are redistributed in this repository only
in so far as they were obtained through the MIT-licensed project above, and any
reuse — particularly commercial reuse — should not assume a permissive licence
applies to them.

---

## saimj7/Social-Distancing-Detection-in-Real-Time

https://github.com/saimj7/Social-Distancing-Detection-in-Real-Time

**Reused:** the YOLOv3 configuration and weights and the Darknet person-detection
approach, retained in `src/detection_legacy.py` for controlled comparison against
the YOLOv8/ByteTrack pipeline. The YOLOv3 model itself originates from
J. Redmon's Darknet project.

```
MIT License

Copyright (c) 2020 Sai Subhakar T

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Ultralytics — YOLOv8n and ByteTrack

https://github.com/ultralytics/ultralytics
Licence text: https://www.gnu.org/licenses/agpl-3.0.txt

**Used:** the YOLOv8n person detector and the ByteTrack tracker, which together
form the deployed detection-and-tracking stage in `src/detection.py`. Declared in
`requirements.txt` as `ultralytics>=8.1`. The package is installed from PyPI; no
part of it is copied into this repository.

**Licence: GNU Affero General Public License v3 (AGPL-3.0).** PyPI records the
package as *"GNU Affero General Public License v3 or later (AGPLv3+)"*.
Ultralytics offers the package under a dual licence: AGPL-3.0 without charge, or a
paid commercial licence. Ultralytics' own licensing guidance lists academic
research and university coursework within the scope of the AGPL-3.0 option.

**What the AGPL requires.** It is a copyleft licence and is materially stronger
than the MIT and BSD terms covering the other components here. It requires the
complete corresponding source of a derivative work to be published under the same
terms, and — unlike the GPL — extends that requirement to software made available
to users over a network rather than distributed to them as a copy.

**Position of this project.** The full source of this project is public in this
repository, and its use is academic, so the practical conditions of the AGPL-3.0
option are met. One point is nonetheless recorded rather than glossed over: this
repository's own `LICENSE` is MIT, and whether a program that imports an AGPL
library forms a derivative work subject to the same terms when distributed is a
question this project does not attempt to resolve. Anyone reusing this code —
especially in a closed, commercial or network-served product — should treat the
detector as AGPL-encumbered and either comply with the AGPL, obtain Ultralytics'
commercial licence, or substitute a detector released under permissive terms.

The AGPL-3.0 text runs to roughly 34,000 characters and is not reproduced here.
It is linked above, and it is distributed with the package itself. Nothing in this
repository redistributes Ultralytics code, so the licence's requirement to convey
a copy of the licence alongside the program is not triggered by this repository.

---

## BAFMD — Bias-Aware Face Mask Detection Dataset

https://github.com/Alpkant/BAFMD

**Reused:** the test-set images and their Pascal VOC annotations, used for the
demographic evaluation reported in `results/mask_eval/mask_eval_bafmd_report.txt`.
Access was requested on 30 June 2026 and granted on 21 August 2026. The dataset is
distributed as image URLs rather than images; 2,055 of 6,264 URLs were no longer
retrievable, yielding 513 of 798 test images. The images are not included in this
repository.

**Two licences apply, and they are not the same.** The MIT licence below covers
the BAFMD *repository* — its annotation format, download tooling and
documentation. It does **not** cover the images, which carry separate terms stated
in the project's README:

- *"Bias Aware Face Mask Detection (BAFMD) dataset is available for non-commercial
  research purposes only."*
- *"By downloading the data you accept the BAFMD dataset Terms of Usage."*
- *"The dataset collected from Twitter, therefore you have to check the terms of
  usage from Twitter to use the images for your purpose as well."*

The use made of the images here — evaluation of a classifier for an academic
dissertation — is non-commercial research. The images are not redistributed in
this repository, and under these terms they could not be.

```
MIT License

Copyright (c) 2021 Alperen Kantarcı

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Datasets used for training-distribution and cross-dataset evaluation

Neither is redistributed in this repository.

- **Baseline dataset** (4,092 images), obtained through
  chandrikadeb7/Face-Mask-Detection above and covered by its MIT licence.
- **Second dataset**, used for the cross-dataset evaluation. Its licence is
  recorded by its host as *Unknown*. It is not redistributed here because the
  rights to do so are undocumented.

---

## Other dependencies

The remaining runtime dependencies declared in `requirements.txt` are all under
permissive licences and impose attribution requirements only, not restrictions on
how this project may be licensed or deployed: NumPy (BSD-3-Clause),
opencv-python (MIT wrapper; the OpenCV library itself as noted above),
imutils (MIT), PyYAML (MIT), Pillow (MIT-CMU/HPND), TensorFlow (Apache-2.0),
tf-keras (Apache-2.0), scikit-learn (BSD-3-Clause), Matplotlib (PSF-based) and
pytest (MIT). They are listed here for completeness; the licences that actually
constrain reuse of this project are the two marked in the summary table.
