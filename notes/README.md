# My AWS Bedrock Learning Notes

These are my personal notes and learnings as I explore AWS Bedrock. I'm documenting my journey to understand how to build generative AI applications using Amazon's fully managed service.



## Bedrock Foundational Models Settings

### LLM Model response Settings
**Temperature, top_k, and top_p** control how an AI model picks the *next word* when generating text.

Suppose the model predicts the next word with probabilities:
- "cat" (40%)
- "dog" (30%)
- "rabbit" (15%)
- "car" (10%)
- "tree" (5%)

- **Temperature** controls creativity. Low temperature picks safer words like **"cat"**, while higher temperature allows more variety like **"rabbit"** or **"car"**.
- **top_k = 2** means the model can only choose from the top 2 words: **"cat"** and **"dog"**.
- **top_p = 0.85** means the model picks from the smallest set of words whose **cumulative probability** reaches 85% → **"cat" (40%) + "dog" (30%) + "rabbit" (15%) = 85%**.

In short: temperature controls *randomness*, top_k limits *how many words*, and top_p limits *how much total probability*.

### Inference Settings

**TODO:** Read more about these options

- **On Demand**
- **Inference Profiles**
- **Provisioned Throughput**


### System Prompt

In AWS Bedrock, a System Prompt is used to set the behavior, role, and rules for the Large Language Model (LLM) before it responds to user input.

```
# System Prompt Example
You are an AWS AI expert.
Explain concepts in simple language for beginners.
Use examples related to AWS services.
```

The model will now consistently respond like an AWS expert and keep explanations beginner-friendly — even if the user doesn’t repeat that instruction.
