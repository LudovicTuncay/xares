## Encoder

`example/whisper/whisper_encoder.py`

---

## Results

| **Domain**      | **Task**                 | **MLP Score** | **KNN Score** |
| --------------- | ------------------------ | ------------: | ------------: |
| **Environment** | Clotho                   |      0.033014 |      0.000000 |
|                 | DESED                    |      0.236192 |      0.000000 |
|                 | ESC-50                   |      0.528500 |      0.191000 |
|                 | FSD18-Kaggle             |      0.252500 |      0.000000 |
|                 | FSD50k                   |      0.194581 |      0.000000 |
|                 | UrbanSound 8k            |      0.684727 |      0.214258 |
| **Music**       | Free Music Archive Small |      0.594286 |      0.421714 |
|                 | GTZAN Genre              |      0.622606 |      0.350323 |
|                 | MAESTRO                  |      0.014546 |      0.000000 |
|                 | NSynth-Instruments       |      0.507568 |      0.205078 |
| **Speech**      | ASV2015                  |      0.955978 |      0.843329 |
|                 | CREMA-D                  |      0.570789 |      0.381720 |
|                 | Fluent Speech Commands   |      0.773003 |      0.032428 |
|                 | LibriCount               |      0.546329 |      0.246154 |
|                 | LibriSpeech-100h         |      0.000000 |      0.000000 |
|                 | LibriSpeech-MF           |      0.976039 |      0.615125 |
|                 | RAVDESS                  |      0.449306 |      0.295833 |
|                 | Speech Commands V1       |      0.930651 |      0.095830 |
|                 | Vocal Imitation          |      0.127834 |      0.016069 |
|                 | VocalSound               |      0.857222 |      0.399109 |
|                 | VoxCeleb1                |      0.387166 |      0.009817 |
|                 | VoxLingua33              |      0.876942 |      0.359851 |

---

## Weighted Averages

| **Category / domain** | **MLP Score** | **KNN Score** |
| --------------------- | ------------: | ------------: |
| Environment           |         0.207 |         0.270 |
| Music                 |         0.524 |         0.270 |
| Speech                |         0.718 |         0.310 |

---

| **All Datasets** | **MLP Score** | **KNN Score** |
| ---------------- | ------------: | ------------: |
| Weighted Average |         0.588 |         0.299 |

| **Public Datasets** | **MLP Score** | **KNN Score** |
| ------------------- | ------------: | ------------: |
| Weighted Average    |         0.588 |         0.299 |

---
