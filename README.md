# Using-the-Open-Source-GPT4All-Model-Locally

## Introduction
The GPT-family models are undoubtedly powerful. However, access to these models' weights and architecture is restricted, and even if one does have access, it requires significant resources to perform any task. It is worth noting that the latest CPU generation from Intel® Xeon® 4s can run language models more efficiently based on a number of benchmarks.

Furthermore, the available APIs are not free to build on top of. These limitations can restrict the ongoing research on Large Language Models (LLMs). The alternative open-source models (like GPT4All) aim to overcome these obstacles and make the LLMs more accessible to everyone.

## How GPT4All works?
It is trained on top of Facebook’s LLaMA model, which released its weights under a non-commercial license. Still, running the mentioned architecture on your local PC is impossible due to the large (7 billion) number of parameters. The authors incorporated two tricks to do efficient fine-tuning and inference. We will focus on inference since the fine-tuning process is out of the scope of this course.

The main contribution of GPT4All models is the ability to run them on a CPU. Testing these models is practically free because the recent PCs have powerful Central Processing Units. The underlying algorithm that helps with making it happen is called Quantization. It basically converts the pre-trained model weights to 4-bit precision using the GGML format.  So, the model uses fewer bits to represent the numbers. There are two main advantages to using this technique:

- Reducing Memory Usage: It makes deploying the models more efficient on low-resource devices.
- Faster Inference: The models will be faster during the generation process since there will be fewer computations.
  
It is true that we are sacrificing quality by a small margin when using this approach. However, it is a trade-off between no access at all and accessing a slightly underpowered model!

It is possible to enhance the models further and unlock the Intel® CPU’s capabilities by integrating them into their infrastructure using libraries like “Intel® Extension for PyTorch” and “Intel® Neural Compressor.” Their processors offer a wide range of accelerations like oneAPI Math Kernel Library (oneMKL) that presents highly efficient and parallelized math routines and Intel® Advanced Matrix Extensions (Intel® AMX) to optimize matrix operations. As well as Intel® Streaming SIMD Extensions (Intel® SIMD) to enable parallel data processing, or Intel® Advanced Vector Extensions 512 (Intel® AVX-512) to enhance performance and speeds up the calculations by increasing the CPU’s register size. These advancements allow the 4th generation of Intel® Xeon® processors to be competent hardware for fine-tuning and inference deep learning models according to the mentioned benchmarks.


# Let’s see in action

## 1. Convert the Model

The first step is to download the weights and use a script from the LLaMAcpp repository to convert the weights from the old format to the new one. It is a required step; otherwise, the LangChain library will not identify the checkpoint file.

We need to download the weights file. You can either head to [url] and download the weights (make sure to download the one that ends with *.ggml.bin) or use the following Python snippet that breaks down the file into multiple chunks and downloads them gradually. The local_path variable is the destination folder.


```

import requests
from pathlib import Path
from tqdm import tqdm

local_path = './models/gpt4all-lora-quantized-ggml.bin'
Path(local_path).parent.mkdir(parents=True, exist_ok=True)

url = 'https://the-eye.eu/public/AI/models/nomic-ai/gpt4all/gpt4all-lora-quantized-ggml.bin'

# send a GET request to the URL to download the file.
response = requests.get(url, stream=True)

# open the file in binary mode and write the contents of the response
# to it in chunks.
with open(local_path, 'wb') as f:
    for chunk in tqdm(response.iter_content(chunk_size=8192)):
        if chunk:
            f.write(chunk)
```


This process might take a while since the file size is 4GB. Then, it is time to transform the downloaded file to the latest format. We start by downloading the codes in the LLaMAcpp repository or simply fork it using the following command. (You need to have the git command installed) Pass the downloaded file to the convert.py script and run it with a Python interpreter.
```
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && git checkout 2b26469
python3 llama.cpp/convert.py ./models/gpt4all-lora-quantized-ggml.bin
```

It takes seconds to complete. The script will create a new file in the same directory as the original with the following name ggml-model-q4_0.bin which can be used in the following subsection.

[Before running the next section’s codes. You need to install the version 0.0.152 of LangChain to be compatible with PyLLaMAcpp package. (pip install -q langchain==0.0.152)]

## 2. Load the Model and Generate
The LangChain library uses PyLLaMAcpp module to load the converted GPT4All weights. Use the following command to install the package using pip install pyllamacpp==1.0.7 and import all the necessary functions. We will provide detailed explanations of the functions as they come up.

```
from langchain.llms import GPT4All
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
```

Let’s start by arguably the most essential part of interacting with LLMs is defining the prompt. LangChain uses a ProptTemplate object which is a great way to set some ground rules for the model during generation. For example, it is possible to show how we like the model to write. (called few-shot learning)

```
template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
```
The template string defines the interaction’s overall structure. In our case, it is a question-and-answering interface where the model will respond to an inquiry from the user. There are two important parts:


- Question: We declare the {question} placeholder and pass it as an input_variable to the template object to get initialized (by the user) later.
- Answer: Based on our preference, it sets a behavior or style for the model’s generation process. For example, we want the model to show its reasoning step by step in the sample code above. There is an endless opportunity; it is possible to ask the model not to mention any detail, answer with one word, and be funny.

  Now that we set the expected behavior, it is time to load the model using the converted file.

```
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = GPT4All(model="./models/ggml-model-q4_0.bin", callback_manager=callback_manager, verbose=True)
llm_chain = LLMChain(prompt=prompt, llm=llm)
```

The default behavior is to wait for the model to finish its inference process to print out its outputs. However, it could take more than an hour (depending on your hardware) to respond to one prompt because of the large number of parameters in the model. We can use the StreamingStdOutCallbackHandler() callback to instantly show the latest generated token. This way, we can be sure that the generation process is running and the model shows the expected behavior. Otherwise, it is possible to stop the inference and adjust the prompt.

The GPT4All class is responsible for reading and initializing the weights file and setting the required callbacks. Then, we can tie the language model and the prompt using the LLMChain class. It will enable us to ask questions from the model using the run() object.

```
question = "What happens when it rains somewhere?"
llm_chain.run(question)
```


```
Question: What happens when it rains somewhere?

Answer: Let's think step by step. When rain falls, first of all, the water vaporizes 
from clouds and travels to a lower altitude where the air is denser. Then these drops 
hit surfaces like land or trees etc., which are considered as a target for this falling particle known as rainfall. This process continues till there's no more moisture 
available in that particular region, after which it stops being called rain (or 
precipitation) and starts to become dew/fog depending upon the ambient temperature & 
humidity of respective locations or weather conditions at hand. Question: What happens 
when it rains somewhere?\n\nAnswer: Let's think step by step. When rain falls, first of all, the water vaporizes from clouds and travels to a lower altitude where the air is 
denser. Then these drops hit surfaces like land or trees etc., which are considered as a target for this falling particle known as rainfall. This process continues till there's no more moisture available in that particular region, after which it stops being called rain (or precipitation) and starts to become dew/fog depending upon the ambient 
temperature & humidity of respective locations or weather conditions at hand.
```

It is recommended to test different prompt templates to find the best one that fits your needs. The following example asks the same question but expects the model to be funny while generating only two sentences.

```
template = """Question: {question}

Answer: Let's answer in two sentence while being funny."""

prompt = PromptTemplate(template=template, input_variables=["question"])
```

```
Question: What happens when it rains somewhere?

Answer: Let's answer in two sentence while being funny. 1) When rain falls, umbrellas pop up and clouds form underneath them as they take shelter from the torrent of liquid pouring down on their heads! And...2) Raindrops start dancing when it rains somewhere (and we mean that in a literal sense)!
```

## Conclusion
We learned about open-source large language models and how to load one in your own PC on Intel® CPU and use the prompt template to ask questions. We also discussed the quantization process that makes this possible. In the next lesson, we will dive deeper and introduce more models while comparing them for different use cases.

In the next lesson, you’ll see a comprehensive guide to the models that can be used with LangChain, along with a brief description of them.
