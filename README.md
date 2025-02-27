# Llama-AVSR: an MLLM Designed for Audio-Visual Speech Recognition

This is the official repository for the paper "[**Large Language Models are Strong Audio-Visual Speech
Recognition Learners**](https://arxiv.org/pdf/2409.12319)", ***U. Cappellazzo***, *M. Kim*, *H. Chen*, *P. Ma*, *S. Petridis*, *D. Falavigna*, *A. Brutti*, *M. Pantic*.

## 📣 News 📣
- ```[2025.03]``` We release the **code** and **ckpts** of Llama-AVSR, along with the camera-ready version of the [paper](https://arxiv.org/abs/2409.12319) 🔥🔥.
- ```[2024.12]``` Our paper has been accepted for publication at **ICASSP** 2025 🚀🚀.
- ```[2024.09]``` We release the [arXiv paper](https://arxiv.org/abs/2409.12319) 🦙.


## Llama-AVSR Overwiew 🔍

**Llama-ASVR** is a Multimodal LLM (MLLM) trained to perform the tasks of ASR, VSR, and AVSR. As such, it comprises three main components: **1)** pre-trained audio and video encoders, **2)** audio/video MLP projector layers, and **3)** a Llama-based LLM, which is parameter-efficiently finetuned via LoRA. **Llama-AVSR** is trained on three different amount of data (30h, 433h, 1756h) and tasks, achieving sota results when tested on the LRS3 dataset. Due to its modularity, **Llama-AVSR** facilitates the seamless integration of various pre-trained encoders and LLMs of different sizes, letting the user choose the configuration based on specific requirements.

<p align="center">
    <img src="assets/main_figure.png" width="50%"> <br>
</p>

## Setup 🛠 
### 1) Installation

Install necessary dependencies: 

```bash
   pip install -r requirements.txt
   cd av_hubert/fairseq
   pip install --editable ./
```

<details>
  <summary><strong>Issues with opencv-python?</strong></summary>
If you encounter issues with opencv-python (e.g., ImportError: libGL.so.1: cannot open shared object file: No such file or directory), pip uninstall opencv-python and pip install opencv-python-headless. This trick solves the issue.

</details>

### 2) Datasets Pre-processing

We rigoroulsy follow auto-avsr [paper](https://arxiv.org/abs/2303.14307) to pre-process the LRS3 and VoxCeleb 2 datasets. All the steps 
to achieve this can be found [here](https://github.com/mpc001/auto_avsr/tree/main/preparation). 

For LRS3, the tree-structure of the directory is like:

```
LRS3  
└───labels
|     lrs3_train_transcript_lengths_seg16s.csv 
|     lrs3_test_transcript_lengths_seg16s.csv 
|       
└───lrs3
    └─── lrs3_text_seg16s
    |     └─── ...
    └───  lrs3_video_seg16s
          └─── ...
```

### 3) Labels files Download

The labels files in LRS3/labels undergo some processing to make it fit Llama-AVSR. For example, we lower the transcription and discard samples whose length is higher than 1s to avoid training instability and peak GPU memory usage. Based on the desired training setting, the processed labels can be accesse below. Once downloaded, they must be moved to dataset/labels subfolder, where dataset can be LRS3 or VoxCeleb2. 




