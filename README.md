# Can LLMs Play the Game They Talk?
## A Linguistic and Behaviorial Analysis of Rationality in [Kuhn Poker](https://sites.math.rutgers.edu/~zeilberg/akherim/PokerPapers/Kuhn1951.pdf)
### 🧠 Built With
![OpenAI](https://img.shields.io/badge/OpenAI-API-blue?logo=openai)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green?logo=langchain)
![NumPy](https://img.shields.io/badge/Numpy-Array-orange?logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-DataFrame-lightgrey?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-yellow?logo=matplotlib)
![SciPy](https://img.shields.io/badge/SciPy-Scientific-blue?logo=scipy)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![tiktoken](https://img.shields.io/badge/tiktoken-Tokenizer-blueviolet)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-red?logo=shap)
### ✅ Status
![Project Status](https://img.shields.io/badge/status-in--development-orange)

> ### Abstract
> This study investigates the game-theoretic rationality of Large Language Models (LLMs)—including GPT-4-Turbo, GPT-4o, and o3-mini—using Kuhn Poker, a simplified game of incomplete information. Across three experiments, we assess whether LLMs exhibit strategic adaptation, probabilistic reasoning, and sensitivity to game-relevant linguistic inputs. **Experiment 1** shows that LLMs stabilize to fixed heuristics in repeated play, suggesting reliance on pretraining priors rather than in-game learning and adaptation. **Experiment 2** finds that GPT-4 variants default to heuristic play unless explicitly guided step-by-step, while o3-mini can independently compute expected values—but exhibits flawed Bayesian updates in certain contexts. This suggests that even reasoning-focused models struggle to fully internalize game structure when it is conveyed purely through natural language, exposing a gap between linguistic understanding and rational inference. **Experiment 3** employs Owen values attribution, an explainable AI (XAI) technique derived from game theory, to identify which input tokens most influence decisions. Results show that models prioritize variables like opponent strategy and held card, but their behavior remains largely deterministic and suboptimal. Despite being able to articulate strategic reasoning in language, LLMs fail to enact them behaviorally. Together, these findings highlight a dissociation between declarative knowledge and procedural execution, suggesting that while LLMs exhibit fragments of strategic competence, their behavior departs from ideal game-theoretic rationality.

### Research Question
- **Can transformer-based LLMs, such as OpenAI’s GPT-4o, GPT-4-Turbo, and o3-mini, exhibit rational behavior in Kuhn Poker?**
- Here, rationality is precisely defined by three measurable abilities:
    - updating beliefs based on the evolving game state and observed actions;
    - accurately computing expected values for various decision options;
    - adapting strategies effectively over repeated interactions.

### Core Discoveries and Contributions
- Provides empirical evidence demonstrating the absence of true online learning/adaptation in current LLMs during repeated play, even when equipped with memory mechanisms (see [plots](/plots/player1_stage1_tokenprobs.png)). This finding suggests that these models do not function as reinforcement learning agents in the traditional sense within prompt-only settings.
- Introduces and empirically supports a concept—“_statistical inertia_”—offering a novel, **non-anthropomorphic** explanation for the observed deterministic LLM behavior in repeated interactions, linking it to the **autoregressive nature of token prediction** through a self-reinforcement loop from game history in prompts.
- Identifies the “_framing problem_”, where the presentation of tasks in natural language, rather than symbolic notation, can impede LLMs’ mathematical and strategic capabilities.
- Uncovers a divide in performance between OpenAI’s `o3-mini` and `GPT-4` variants, highlighting a fundamental distinction between _declarative knowledge_ (the ability to articulate game-theoretic concepts) and _procedural execution_ (the capacity to apply them dynamically in a game context).
- Innovatively applies **SHAP (SHapley Additive exPlanations) with Owen value decomposition** ([Lundberg & Lee, 2017](https://arxiv.org/abs/1705.07874)) to provide valuable, albeit still constrained, insights into the **black-box nature of LLM decision-making**, offering a diagnostic lens into how prompt components may influence choices.
> For a concise walk-through of the study, please refer to the [thesis defense slides](slides/thesis-defense-slides.pdf)
