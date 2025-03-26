# Contribute to Open Unlearning

Everyone is welcome to contribute, and every contribution is valued. While code contributions are important, there are many other ways to support the community, answering questions, assisting others, and improving documentation are all immensely valuable.

You can also help by spreading the word! If you find this project useful, consider giving the repository a ⭐️ to show your support.

This guide was heavily inspired by the awesome [transformers](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md) guide to contributing.

## Ways to contribute

There are several ways you can contribute to Open Unlearning:

* Fix issues with the existing code.
* Submit issues related to bugs or desired new features.
* Support new components (models, datasets, collator etc).
* Implement new unlearning methods.
* Implement new evaluations.
* Contribute to the documentation.

> All contributions are equally valuable to the community. 🥰

## Fixing issues

If you notice an issue with the existing code and have a fix in mind, feel free to [start contributing](#create-a-pull-request) and open a Pull Request!

## Submitting a bug-related issue or feature request

Do your best to follow these guidelines when submitting a bug-related issue or a feature
request. It will make it easier for us to come back to you quickly and with good
feedback.

### Did you find a bug?

Before you report an issue, we would really appreciate it if you could **make sure the bug was not
already reported** (use the search bar on GitHub under Issues). Your issue should also be related to bugs in the library itself, and not your code. If you're unsure whether the bug is in your code or the library, please ask in the [discord](https://discord.gg/v4aYBZsW) first. This helps us respond quicker to fixing issues related to the library versus general questions.

Once you've confirmed the bug hasn't already been reported, please include the following information in your issue so we can quickly resolve it:

* A short, self-contained, code snippet that allows us to reproduce the bug.
* The *full* traceback if an exception is raised.
* The hardware used to run the experiment, including specifications such as the number and type of GPUs etc.
* Attach any other additional information, like screenshots, you think may help.


### Do you want a new feature?

If there is a new feature you'd like to see in Open Unlearning, please open an issue and describe:

1. What is the *motivation* behind this feature? Is it related to a problem or frustration with the library? Is it something you worked on and think it could benefit the community?

   Whatever it is, we'd love to hear about it!

2. Describe your requested feature in as much detail as possible. The more you can tell us about it, the better we'll be able to help you.
3. Provide a *code snippet* that demonstrates the features usage.
4. If the feature is related to a paper, please include a link.

## Do you want to support new components


Adding a new component listed below requires defining a new class, registering it, and creating a configuration file. Learn more about adding new components in [`docs/components.md`](docs/components.md).

1. [Trainer](components#trainer) - Algorithm used in LLM training or unlearning
2. [Dataset](components#dataset) - Dataset class for preprocessing raw data
3. [Evaluation Metric](components#evaluation-metric) - Metric class implementing model evaluation
4. [Benchmark](components#benchmark) - Suite combining multiple evaluation metrics
5. [Model](components#model) - LLM used in unlearning
6. [Collator](components#collator) - Handles data collation logic
7. [Experiment](components#experiment) - Combines components into a final experiment config
8. 

## Contributing a New Unlearning Method


We greatly appreciate your implementation contributions to unlearning methods, as you understand the intricacies best than anyone. If you encounter any challenges, feel free to discuss your approach in our [discord](https://discord.gg/v4aYBZsW) group, we are always here to help!

Steps to contribute for new unlearning method

### 1. Implement an Unlearning Trainer

Refer to our [Trainer implementation guide](components.md#trainer) to ensure your method integrates well with our framework.

### 2. Run and Tune Your Method on Relevant Benchmarks

- Once implemented, evaluate your method on applicable benchmarks using the best possible parameters.
- Create a folder [`community/methods/<YOUR_METHOD>`](../community/methods) and include a README file in it, explaining the method details, hyper-parameters, strategy/logic for selecting the best model for unlearning etc.
- Include a bash script `run.sh` with the exact bash command needed to replicate your results.


### 3. Update leaderboard and share model

Don't forget to add your results to the [leaderboard](results.md) and, if possible, upload your unlearned model to Hugging Face for broader accessibility and reproducibility.

```bash
pip install huggingface_hub
huggingface-cli login

huggingface-cli repo create {benchmark}-{model}-{datasplit}-{method}
cd <CHECKPOINT_DIR>

git init
git remote add origin https://huggingface.co/<username>/{benchmark}-{model}-{datasplit}-{method}
git add .
git commit -m "Initial commit"
git push origin main
```


## Contributing a New Unlearning Benchmark

Evaluating LLM unlearning is essential for assessing the effectiveness of different unlearning methods. While various benchmarks and metrics exist, identifying the most suitable ones for capturing the nuances of unlearning remains an open challenge.

Your contributions toward defining or improving evaluation methods can significantly advance unlearning research. By proposing reliable benchmarks, you help ensure that unlearning methods are both effective and aligned with real-world requirements.

- To add a new unlearning metric, refer to our [Metric Implementation Guide]((components.md#evaluation-metric).).
- To integrate new datasets and models, follow our [Components Guide](components.md).

### Steps to contribute for new unlearning benchmark
1. **Prepare Datasets & Models** – Create your dataset and train models to generate fine-tuned or retained models.
2. **Define a New Benchmark** (if needed) – Follow the [Benchmark Guide]((components.md#benchmark)) to implement a new evaluation benchmark.
3. **Run and Tune Baseline Methods** – Evaluate existing unlearning methods on your benchmark and optimize them.
4. **Document & Share Findings** – Provide detailed steps for reproduction in [`community/benchmarks/<YOUR_BENCHMARK>`](../community/benchmarks).

## Do you want to add documentation?

We're always looking for improvements to the documentation that make it more clear and accurate. Please let us know how the documentation can be improved such as typos and any content that is missing, unclear or inaccurate. We'll be happy to make the changes or help you make a contribution if you're interested!

## Create a Pull Request

<!-- Let's first fork the transformers repo on github. Once it's done you can clone your fork and install transformers in our environment: -->

Before writing any code, we strongly advise you to search through the existing PRs or
issues to make sure nobody is already working on the same thing. If you are
unsure, it is always a good idea to open an issue to get some feedback.

You will need basic `git` proficiency to contribute to
open-unlearning. While `git` is not the easiest tool to use, it has the greatest
manual. Type `git --help` in a shell and enjoy! If you prefer books, [Pro
Git](https://git-scm.com/book/en/v2) is a very good reference.

Follow the steps below to start contributing:

1. Fork the [repository](https://github.com/huggingface/transformers) by
   clicking on the **[Fork](https://github.com/huggingface/transformers/fork)** button on the repository's page. This creates a copy of the code
   under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

   ```bash
   git clone git@github.com:<your Github handle>/open-unlearning.git
   cd open-unlearning
   git remote add upstream https://github.com/locuslab/open-unlearning.git
   ```

3. You can work on the forked main branch or create a new branch to hold your development changes:

   ```bash
   git checkout -b a-descriptive-name-for-my-changes
   ```

4. Set up a development environment by running the following command in a virtual environment:

   ```bash
   conda create -n unlearning python=3.11
   conda activate unlearning
   pip install .[dev]
   pip install --no-build-isolation flash-attn==2.6.3
   ```

5. Develop the features in your fork/branch.

   As you work on your code, you should make sure the code is linted and formatted correctly.

   Open Unlearning relies on `ruff` to lint & format its source code
   consistently. After you make changes, to check the quality of code, run

   ```bash
   make quality
   ```

   If you prefer to apply the style corrections:

   ```bash
   make style
   ```

   Once you're happy with your changes, add the changed files with `git add` and
   record your changes locally with `git commit`:

   ```bash
   git add modified_file.py
   git commit
   ```

   Please remember to write [good commit
   messages](https://chris.beams.io/posts/git-commit/) to clearly communicate the changes you made!

   To keep your copy of the code up to date with the original
   repository, rebase your branch on `upstream/branch` *before* you open a pull request or if requested by a maintainer:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

   Push your changes to your branch:

   ```bash
   git push -u origin a-descriptive-name-for-my-changes
   ```

   If you've already opened a pull request, you'll need to force push with the `--force` flag. Otherwise, if the pull request hasn't been opened yet, you can just push your changes normally.

6. Now you can go to your fork of the repository on GitHub and click on **Pull Request** to open a pull request. Make sure you tick off all the boxes on our [checklist](#pull-request-checklist) below. When you're ready, you can send your changes to the project maintainers for review.

7. It's ok if maintainers request changes, it happens to our core contributors
   too! So everyone can see the changes in the pull request, work in your local
   branch and push the changes to your fork. They will automatically appear in
   the pull request.

### Pull request checklist

☐ The pull request title should summarize your contribution.<br>
☐ If your pull request addresses an issue, please mention the issue number in the pull
request description to make sure they are linked (and people viewing the issue know you
are working on it).<br>
☐ To indicate a work in progress please prefix the title with `[WIP]`. These are
useful to avoid duplicated work, and to differentiate it from PRs ready to be merged.<br>
☐ Make sure existing tests pass.<br>
☐ Make methods having informative docstrings.<br>
