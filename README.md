## Content We Make Public in This Project

The prompts and related parameters used for the LLM, the MSA terminal code, and the complete dataset, including AI-enhanced texts and the extracted features.

## Acoustic Feature Discretization

For the acoustic semantic enhancement pipeline, continuous acoustic descriptors were discretized into categorical levels before being converted into textual descriptions. For each dataset, the 33rd and 66th percentiles were estimated exclusively from the training split and used as fixed thresholds for discretizing loudness, loudness dynamics, pitch variability, voicing ratio, speaking-rate proxy, and utterance duration. The same training-derived thresholds were then applied unchanged to the validation and test sets.

The resulting categories were low / mid / high for most descriptors and short / mid / long for duration. Pitch trend was handled separately using fixed thresholds on the F0 slope: values below -10 Hz/s were labeled as falling, values between -10 and 10 Hz/s as flat, and values above 10 Hz/s as rising. When no voiced frames were detected (voiced_ratio = 0), pitch-related descriptors such as F0 variability and pitch slope were treated as missing rather than assigned to a valid category.

For reproducibility, the train-derived thresholds were:

Descriptor	MOSI q33 / q66	MOSEI q33 / q66
Loudness (rms_mean)	0.01239 / 0.03418	0.02308 / 0.04617
Loudness dynamics (rms_range_p95_p05)	0.03211 / 0.09291	0.06311 / 0.12043
Pitch variability (f0_std_hz)	17.6015 / 33.1999	19.1222 / 35.8990
Voicing ratio (voiced_ratio)	0.38323 / 0.56435	0.26038 / 0.46628
Speaking-rate proxy (rate_proxy_peaks_per_sec)	4.41176 / 4.96036	4.01203 / 4.77313
Duration (duration_sec)	2.35 / 4.2478	8.27 / 12.27

## LLM-based Audio Description

The following prompt is used for the audio modality to generate
acoustic prosody descriptions from the extracted audio features.

No emotion labels, annotations, or transcript content are provided
to the LLM during audio description generation.

### Model Configuration
- Model: GPT-4o-mini
- Temperature: 0.2
- Max tokens: 320
- Seed: 42

Due to potential differences in seed handling and minor variations across execution environments, the generated content may differ slightly across runs.


## Data Availability

We have shared the processed and extracted data on Baidu Netdisk under the name DATA-for-Semer-MSA.
Link: https://pan.baidu.com/s/1Ii819YVgUUVDiE41AMF4DQ
Extraction code: er41 

To support the reproducibility of our experiments, the processed PKL files used in this study will be made available via Baidu Netdisk. If Baidu Netdisk is not accessible in your region, we will also provide the corresponding experiment logs to facilitate verification of the reported results.

The provided PKL files are made available solely for the purpose of peer review and reproducibility assessment of this submission. They must not be used, copied, redistributed, or repurposed for any other purpose without explicit permission from the authors.

## Data Preprocessing Notes
During data preprocessing, we manually inspected and corrected a small number of samples in CMU-MOSI and CMU-MOSEI that showed clear anomalies. In CMU-MOSEI, approximately 20-50 samples were corrected or removed due to issues such as corrupted files, abnormal durations, or inconsistent information. Two additional samples in CMU-MOSI underwent similar processing.

In addition, a small number of auxiliary annotation fields were corrected. These fields were not used for model training, validation, or final performance evaluation. Since these cleaning steps were performed at an early stage of the project, a complete sample-level modification log is no longer available. We therefore report the general processing principles and approximate scale here rather than providing a potentially inaccurate retrospective list.


