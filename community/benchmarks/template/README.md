# Your benchmark

List of Authors: 

Link to Paper: 


# Introduction

Provide a concise summary of your benchmark, highlighting its key contributions and how it differs from existing benchmarks. Avoid using images to keep the repository size manageable.


# Datasets

- [ ] Provide a link to the Hugging Face datasets for easy access.
- [ ] Use a clear and consistent naming convention for all dataset splits.

# Models

- [ ] Upload any fine-tuned or retain models for unlearning to your Hugging Face account and provide the links.
- [ ] Explain why these models were chosen and how they are relevant to the benchmark.

# Baselines & Results

Discuss the baselines used and their performance results.


## Setup
Please include the experimental setup for the baselines

- [ ] **Hyperparameters & Search Space:** Specify key hyperparameters and their search ranges.
- [ ] **Computational Setup:** Mention the type and number of GPUs used.
- [ ] **DeepSpeed Configuration:** If any modifications were made to the default DeepSpeed config, specify them here.
- [ ] **Other Relevant Details:** Any additional setup details crucial for reproducing your method.

To replicate your results, provide a `run.sh` script that contains all necessary commands to reproduce the final results. Ensure the script is well-documented.


# Citation


If you use this work, please cite:

```bibtex

<INCLUDE YOUR CITATION>

@misc{openunlearning2025,
  title={OpenUnlearning: A Unified Framework for LLM Unlearning Benchmarks},
  author={Dorna, Vineeth and Mekala, Anmol and Zhao, Wenlong and McCallum, Andrew and Kolter, J Zico and Maini, Pratyush},
  year={2025},
  howpublished={\url{https://github.com/locuslab/open-unlearning}},
  note={Accessed: February 27, 2025}
}
```