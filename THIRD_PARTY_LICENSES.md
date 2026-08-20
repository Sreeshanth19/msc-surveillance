# Third-Party Licences

This project reuses components from the following MIT-licensed projects.
Their licence texts are reproduced in full below as required by the MIT terms.

---

## chandrikadeb7/Face-Mask-Detection

https://github.com/chandrikadeb7/Face-Mask-Detection

**Reused:** the trained `mask_detector.model` MobileNetV2 classifier; the
res10 SSD face-detector assets (`deploy.prototxt`, `res10_300x300_ssd_iter_140000.caffemodel`);
the 4,092-image mask dataset; and the face-detection/mask-classification
procedure adapted in `src/mask_classifier.py`.

**Note:** the face-detector assets originate from OpenCV's `samples/dnn/face_detector`
and were obtained via this project. `models/deploy.prototxt` is byte-identical
to the copy distributed there.

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
