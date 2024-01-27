# Using-the-Open-Source-GPT4All-Model-Locally

## Introduction
The GPT-family models which we covered earlier are undoubtedly powerful. However, access to these models' weights and architecture is restricted, and even if one does have access, it requires significant resources to perform any task. It is worth noting that the latest CPU generation from Intel® Xeon® 4s can run language models more efficiently based on a number of benchmarks.

Furthermore, the available APIs are not free to build on top of. These limitations can restrict the ongoing research on Large Language Models (LLMs). The alternative open-source models (like GPT4All) aim to overcome these obstacles and make the LLMs more accessible to everyone.

## How GPT4All works?
It is trained on top of Facebook’s LLaMA model, which released its weights under a non-commercial license. Still, running the mentioned architecture on your local PC is impossible due to the large (7 billion) number of parameters. The authors incorporated two tricks to do efficient fine-tuning and inference. We will focus on inference since the fine-tuning process is out of the scope of this course.

The main contribution of GPT4All models is the ability to run them on a CPU. Testing these models is practically free because the recent PCs have powerful Central Processing Units. The underlying algorithm that helps with making it happen is called Quantization. It basically converts the pre-trained model weights to 4-bit precision using the GGML format.  So, the model uses fewer bits to represent the numbers. There are two main advantages to using this technique:

1- Reducing Memory Usage: It makes deploying the models more efficient on low-resource devices.
2- Faster Inference: The models will be faster during the generation process since there will be fewer computations.
