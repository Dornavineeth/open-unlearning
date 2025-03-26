# Your method name / Paper Title

List of Authors: 

Link to Paper: 


# Introduction

Provide a concise summary of your method. Explain its key contributions and how it differs from existing approaches. Please avoid using images to keep repository size manageable.


# Setup

Please include the experimental setup such as

- [ ] **Hyperparameters & Search Space:** Specify key hyperparameters, their search ranges, total trials etc.
- [ ] **Computational Setup:** Mention the type and number of GPUs used.
- [ ] **DeepSpeed Configuration:** If any modifications were made to the default DeepSpeed config, specify them here.
- [ ] **Other Relevant Details:** Any additional setup details crucial for reproducing your method.

# Results

To replicate your results, provide a `run.sh` script that contains all necessary commands to reproduce the final results. Ensure the script is well-documented.

We would appreciate it if you could upload the final unlearned model along with its `evals` folder to Hugging Face and provide the link here.

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