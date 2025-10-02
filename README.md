# Can LLMs Play the Game They Talk?
## A Linguistic and Behaviorial Analysis of Rationality in Kuhn Poker
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
