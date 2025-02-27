# Llama-AVSR: an MLLM Designed for Audio-Visual Speech Recognition

This is the official repository for the paper "[**Large Language Models are Strong Audio-Visual Speech
Recognition Learners**](https://arxiv.org/pdf/2409.12319)", ***U. Cappellazzo***, *M. Kim*, *H. Chen*, *P. Ma*, *S. Petridis*, *D. Falavigna*, *A. Brutti*, *M. Pantic*.

## 📣 News 📣
- ```[2025.03]``` We release the **code** and **ckpts** of Llama-AVSR, along with the camera-ready version of the [paper](https://arxiv.org/abs/2409.12319) 🔥🔥.
- ```[2024.12]``` Our paper has been accepted for publication at **ICASSP** 2025 🚀🚀.
- ```[2024.09]``` We release the [arXiv paper](https://arxiv.org/abs/2409.12319) 🦙.


## Llama-AVSR Overwiew 🔍

**Llama-ASVR** is a Multimodal LLM (MLLM) trained to perform the tasks of ASR, VSR, and AVSR. As such, it comprises three main components: **1)** pre-trained audio and video encoders, **2)** audio/video MLP projector layers, and **3)** a Llama-based LLM, which is parameter-efficiently finetuned via LoRA. **Llama-AVSR** is trained on three different amount of data (30h, 433h, 1756h) and tasks, achieving sota results when tested on the LRS3 dataset. Due to its modularity, **Llama-AVSR** facilitates the seamless integration of various pre-trained encoders and LLMs of different sizes, letting the user choose the configuration based on specific requirements.

