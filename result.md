| Exp ID | Train Data   | Model                      | 주요 설정                    | Best Epoch | Best Threshold | Val Macro-F1 | F1 ORIGINAL | F1 RERECORDED | Train Time | Max VRAM | 비고             |
| ------ | ------------ | -------------------------- | ------------------------ | ---------: | -------------: | -----------: | ----------: | ------------: | ---------: | -------: | -------------- |
| A1     | DLC          | VideoMAEv2-B               | 16f, 224, stride {1,2,4} |6|0.9608|0.9704|0.9655|0.9752|            |          | Video baseline |
| A2     | DLC          | Bayar+R18                  | native patch 256         |4|0.1653|0.9902|0.9888|0.9916|            |          | F1             |
| A3     | DLC          | Chromaticity               | native patch 256         |5|0.4794|0.9212|0.9091|0.9333|            |          | F3             |
| A4     | DLC          | Frequency                  | FFT / moiré              |8|0.3084|0.9409|0.9318|0.9500|            |          | F4             |
| A5     | DLC          | LC&DF-inspired             | chroma + freq            |6|0.1962|1.0000|1.0000|1.0000|            |          | F5 main        |
| A6     | DLC          | CDC                        | native patch 256         |4|0.5000|0.9705|0.9663|0.9748|            |          | F2             |
| A7     | DLC          | V-JEPA 2.1-B               | 16f, 384                 |2|0.5000|1.0000|1.0000|1.0000|            |          | Video alt      |
| A8     | DLC          | Best Video + Best Forensic | late fusion α=           |          — |                |              |             |               |          — |        — | Fusion         |
| B1     | DLC + CCD-OR | VideoMAEv2-B               | same as A1               |            |                |              |             |               |            |          | Δ vs A1        |
| B2     | DLC + CCD-OR | LC&DF-inspired             | same as A5               |            |                |              |             |               |            |          | Δ vs A5        |
| B3     | DLC + CCD-OR | Best forensic #2           | same setting             |            |                |              |             |               |            |          | optional       |
| B4     | DLC + CCD-OR | V-JEPA 2.1-B               | same as A7               |            |                |              |             |               |            |          | Δ vs A7        |
| B5     | DLC + CCD-OR | Best Video + Best Forensic | late fusion α=           |          — |                |              |             |               |          — |        — | Fusion         |


| Fusion | Video | Forensic | α(Video) | Threshold | Val Macro-F1 | Δ vs Best Single | Prediction Corr |
| ------ | ----- | -------- | -------: | --------: | -----------: | ---------------: | --------------: |
| FUS-A1 |       |          |          |           |              |                  |                 |
| FUS-A2 |       |          |          |           |              |                  |                 |
| FUS-B1 |       |          |          |           |              |                  |                 |
