| Exp ID | Train Data   | Model                      | 주요 설정                    | Best Epoch | Best Threshold | Val Macro-F1 | F1 ORIGINAL | F1 RERECORDED | Train Time | Max VRAM | 비고             |
| ------ | ------------ | -------------------------- | ------------------------ | ---------: | -------------: | -----------: | ----------: | ------------: | ---------: | -------: | -------------- |
| A1     | DLC          | VideoMAEv2-B               | 16f, 224, stride {1,2,4} |            |                |              |             |               |            |          | Video baseline |
| A2     | DLC          | Bayar+R18                  | native patch 256         |            |                |              |             |               |            |          | F1             |
| A3     | DLC          | Chromaticity               | native patch 256         |            |                |              |             |               |            |          | F3             |
| A4     | DLC          | Frequency                  | FFT / moiré              |            |                |              |             |               |            |          | F4             |
| A5     | DLC          | LC&DF-inspired             | chroma + freq            |            |                |              |             |               |            |          | F5 main        |
| A6     | DLC          | CDC                        | native patch 256         |            |                |              |             |               |            |          | F2             |
| A7     | DLC          | V-JEPA 2.1-B               | 16f, 384                 |            |                |              |             |               |            |          | Video alt      |
| A8     | DLC          | Best Video + Best Forensic | late fusion α=           |          — |                |              |             |               |          — |        — | Fusion         |
| B1     | DLC + CCD-OR | VideoMAEv2-B               | same as A1               |            |                |              |             |               |            |          | Δ vs A1        |
| B2     | DLC + CCD-OR | LC&DF-inspired             | same as A5               |            |                |              |             |               |            |          | Δ vs A5        |
| B3     | DLC + CCD-OR | Best forensic #2           | same setting             |            |                |              |             |               |            |          | optional       |
| B4     | DLC + CCD-OR | V-JEPA 2.1-B               | same as A7               |            |                |              |             |               |            |          | Δ vs A7        |
| B5     | DLC + CCD-OR | Best Video + Best Forensic | late fusion α=           |          — |                |              |             |               |          — |        — | Fusion         |
