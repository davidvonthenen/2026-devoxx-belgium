# Devoxx Belgium 2026 - Underneath the Black Marker: AI Techniques for Detecting and Reversing Redactions

Welcome to the landing page for the session `Underneath the Black Marker: AI Techniques for Detecting and Reversing Redactions` at `Devoxx Poland 2026`.

## What to Expect

This repo intends to provide an introduction to:

- Building a Small Language Model (SLM) From scratch
- Provide a guide for Fine-tuning and Quantization
- Provide a guide for Distillation Training

> **IMPORTANT:** Do not actually use any of these models for production. There are already a **TON** of models that do this exact thing. (Check out HuggingFace) Definitely, use those models over these. 

## Hardware Prerequisites

Demos 1 to 8 should work on any laptop.

Demo 9:
- (Optional) Training will require an H100. There is simply no getting around that.
- If you don't have access to this kind of hardware, you can at least download the pre-built models for inference.

## Software Prerequisites

- A Linux or Mac-based Developer's Laptop 
  - Windows Users should use a VM or Cloud Instance
- Python Installed: version 3.12 or higher
- (Recommended) Using a miniconda or venv virtual environment
- Basic familiarity with shell operations

## Participation Options

There are 3 separate demo projects:

- [Demo 1: Copy and Paste](./demos/1_obfuscate_copypaste)
- [Demo 2: Programmatically Copy and Paste](./demos/2_obfuscate_prog)
- [Demo 3: PDF Revisions](./demos/3_revisions)
- [Demo 4: Manipulation PDF Metadata](./demos/4_metadata)
- [Demo 5: Reconstructing Email Attachments](./demos/5_email_mime)
- [Demo 6: Look at `Depixelation` Project](./demos/6_depixelization)
- [Demo 7: Look at `unpixel` Project](./demos/7_unpixel)
- [Demo 8: Look at `DeepMosaics` Project](./demos/8_deepmosaics)
- [Demo 9: My Implementation](./demos/9_mine/README.md)
  - Dataset:
    - PRIMARY DOWNLOAD: https://drive.google.com/file/d/1zWLFMgh6aOUxd9LYG3wrUzRi7Cn2DkkP/view?usp=drive_link
    - SECONDARY DOWNLOAD: https://drive.google.com/file/d/10xPD9VFDGBczoduFavbpG1yPb6oV6T1E/view?usp=drive_link
  - Safetensor Checkpoint for Inference:
    - PRIMARY DOWNLOAD: https://drive.google.com/file/d/1Syjiawhi2HJzTNkmbl0IrTo7Ro5vMBNf/view?usp=drive_link
    - SECONDARY DOWNLOAD: https://drive.google.com/file/d/1k6HVrgLuRmsjE7GEGH5TDNi-R5beQGXI/view?usp=drive_link
- [Bonus: Try This on Your Own!](./demos/bonus)

The instructions and purpose for each demo is contained within their respective folders.
