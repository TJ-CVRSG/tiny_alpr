# Tiny ALPR

This is the official repository for the paper "Ultra-Lightweight ALPR System for Microcontrollers: A Cost-Effective and Energy-Efficient Solution".

Author: Shaoming Zhang, Yan Tang, Weijia Liu, Jianmei Wang

## Introduction

In the realm of intelligent transportation systems, Automatic License Plate Recognition (ALPR) is integral, yet is often hindered by heavy computational loads and massive memory footprints. Existing methods either enhance recognition accuracy through complex models or minimize resource requirements by utilizing low-performance models and low-resolution inputs, both of which pose challenges in real-world applications. To address these issues, this paper introduces a novel ultra-lightweight ALPR system specifically designed for deployment on microcontrollers, offering a cost-effective and energy-efficient solution for large-scale applications. We simplify the process by decoupling ALPR into three sub-tasks: detection, localization, and recognition. For each sub-task, specific optimization strategies such as RLE-based keypoints localization, adaptive data augmentation, and multi-voting recognition mechanism are applied. This proposed approach naturally mitigates the issues of large perspective transformations frequently encountered in real-world settings. As a supplementary contribution, we introduce and release the TJLP dataset. Practical application and superior performance of our method have been demonstrated on resource-limited devices like RV1106 (ARM Cortex-A7 with a 0.5TOPS NPU), achieving an inference speed of 46 FPS with a power consumption of 629 mW. Notably, our fully optimized model exhibits a recognition accuracy of 99.23% in real-world environments.

## ALPR Demo

You can view the model we trained on CCPD by running `pipeline.py`.

## TJLP Dataset

You can download the TJLP dataset from: [Google Drive](https://drive.google.com/file/d/1EthpC1Q4yecENktQxeGf_jP12slqZF19/view?usp=drive_link). You need to contact us to get the password.

Following is the template to apply for the password:

```plain
I am a researcher from XXX. I am interested in your paper "Ultra-Lightweight ALPR System for Microcontrollers: A Cost-Effective and Energy-Efficient Solution". I would like to apply for the password of the TJLP dataset.

My information is as follows:

Name: XXX
Affiliation: XXX
Email: XXX
Date: XXX
```

## Acknowledgment

If you find this project useful in your research, please consider citing our paper.
