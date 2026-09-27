# OCR models

`text-detection.rten` and `text-recognition.rten` are the pre-trained
[ocrs](https://github.com/robertknight/ocrs) models by Robert Knight,
published at <https://huggingface.co/robertknight/ocrs> and downloaded
unmodified from `https://ocrs-models.s3-accelerate.amazonaws.com/`.

They are licensed under
[Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
(CC BY-SA 4.0) and were trained on the
[HierText](https://github.com/google-research-datasets/hiertext) dataset
(CC BY-SA 4.0) and synthetic data.

They are compiled into quillrag only when the `ocr` Cargo feature is enabled.
The rest of quillrag remains MIT-licensed; the ShareAlike terms apply to the
models themselves and any modified versions of them.

| File | SHA-256 |
|---|---|
| text-detection.rten | `f15cfb56bd02c4bf478a20343986504a1f01e1665c2b3a0ad66340f054b1b5ca` |
| text-recognition.rten | `e484866d4cce403175bd8d00b128feb08ab42e208de30e42cd9889d8f1735a6e` |
