# Learning Artificial Intelligence  

Here I will be posting resources and projects related to my journey of learning artificial intelligence.  

Originally, I started my Computer Science journey with web development.  

---

### 1. DChat Application  
I built an application called **DChat**, which is a web application where users can upload PDFs and ask questions related to them.  
Here, I encountered LLM models like **Gemini 2.0 Flash** and **Mistral AI**, which I used in my application. From there, I got curious about how these models were able to give such great responses. After this, I wanted to know how LLMs work internally.  

- **Demo link:** [LinkedIn Demo](https://www.linkedin.com/posts/thinley-lama-842631252_ai-chatbot-rag-activity-7258469838874910721-ro5A?utm_source=share&utm_medium=member_desktop&rcm=ACoAAD5ipnIBKx11QjzcH6rujsE2BXJ-D3FhIOU)  
- **GitHub link:** [DChat Repository](https://github.com/blockx3/dchat)  

While building this RAG-based chatbot, I spent a large amount of time reading **LangChain (Python and JavaScript) documentation** and implementing concepts.  

Here I was introduced to:  
- **Embedding data** – converting words into numerical form  
- **Vector databases** (Postgres + PGVector)  

---

### 2. Article: How LLMs Work  
I found a great article that explained **how LLMs work**. This blog gave me an idea of the internal workings of LLMs.  
- [Read here](https://medium.com/data-science-at-microsoft/how-large-language-models-work-91c362f5b78f)  

---

### 3. Andrej Karpathy – Neural Networks: Zero to Hero  
I enrolled in the **Zero to Hero playlist** by Andrej Karpathy.  

I started with:  
- **Let’s build GPT: from scratch, in code, spelled out**  
  - [YouTube Link](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=21s)  

We built GPT following the paper *“Attention is All You Need”*.  

Here I learned about:  
- Tokenization, train/val split  
- Self-attention  
- Transformer, feedforward  
- Multi-headed self-attention  

It was a hands-on learning experience where I coded along in Colab using **PyTorch**. I was also introduced to the PyTorch documentation. Honestly, I didn’t understand everything, but I was able to get the intuition. I spent 3–4 days completing the 2-hour video.  

I also did hands-on work with:  
- **The spelled-out intro to neural networks and backpropagation: building micrograd**  
  - Learned *manual backpropagation*, derivation, and the importance of mathematics in neural networks.  

Other useful resources:  
- [Video 1](https://www.youtube.com/watch?v=PaCmpygFfXo&t=150s)  
- [Video 2](https://www.youtube.com/watch?v=TCH_1BHY58I&t=1243s)  

In my opinion, these videos are great resources but require some prior knowledge to fully follow along.  

---

### 4. Project: Handwritten Digit Recognition  
In this project, I used the **MNIST dataset** consisting of handwritten digit images and created a **Convolutional Neural Network (CNN)** model.  

- Trained the CNN model on the dataset, and the model was able to predict handwritten digits.  
- Integrated the model with **Streamlit**, where users can upload an image and get the predicted digit.  

- **GitHub link:** [Handwritten Digit Recognition](https://github.com/thinley4/Handwritten-digit-recognition)  

---

### 5. Stanford YouTube Videos  
Some great resources available online:  
- [Stanford CS229 | Machine Learning | Building Large Language Models (LLMs)](https://www.youtube.com/watch?v=9vM4p9NN0Ts&t=5244s)  
- [Introduction to Convolutional Neural Networks for Visual Recognition](https://www.youtube.com/watch?v=vT1JzLTH4G4&t=3s)  
- [Image Classification](https://www.youtube.com/watch?v=OoUX-nOEjG0)  
- [Loss Functions and Optimization](https://www.youtube.com/watch?v=h7iBpEHGVNc&t=853s)  

---

### 6. Machine Learning Specialization – Andrew Ng (Autumn 2018)  
- [Lecture Playlist](https://www.youtube.com/watch?v=jGwO_UgTS7I&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU)  
- [GitHub Code](https://github.com/greyhatguy007/Machine-Learning-Specialization-Coursera)  

I have completed **4 lectures** so far and look forward to completing the rest soon.  

---

### 7. PyTorch  
I started learning **PyTorch** and spent about a week jumping between tutorials and the documentation.  

Resources I followed:  
- [Deep Learning with PyTorch – Full Course](https://www.youtube.com/watch?v=V_xro1bcAuA&t=48258s)  
- [Learn PyTorch](https://www.learnpytorch.io/)  
- [PyTorch Official Documentation](https://docs.pytorch.org/docs/stable/index.html)  

---

### 8. Project: AI Agent (Gmail Agent)  
I built a **Gmail Agent** project using **LangChain**, **Gemini model**, and **Streamlit**.  
Users can simply provide information, and the agent assists accordingly.  

- **Demo link:** [LinkedIn Demo](https://www.linkedin.com/posts/thinley-lama-842631252_ai-aiagent-activity-7299138238336708610-Wxl6?utm_source=share&utm_medium=member_desktop&rcm=ACoAAD5ipnIBKx11QjzcH6rujsE2BXJ-D3FhIOU)  
- **GitHub link:** [Gmail Agent Repository](https://github.com/thinley4/Gmail-Agent)  

---

### 9. Sebastian Raschka – LLMs from Scratch  
Among all resources, this one is my favorite. I got the chance to **build LLMs from scratch** with step-by-step explanations and code.  

What I learned:  
- Tokenization, token IDs, special context tokens, byte pair encoding  
- Data sampling with a sliding window  
- Token embeddings, positional encoding  
- Self-attention mechanism, causal attention mask, multi-head attention  
- Transformer blocks with layer normalization, GELU activations, residual connections  
- Implementing GPT model and text generation  
- Loss functions (cross-entropy, perplexity), training/validation losses  
- Saving/loading pretrained weights  
- Finetuning LLMs for tasks like spam classification  
- Supervised instruction finetuning on datasets  

Resources:  
- [GitHub Repository](https://github.com/rasbt/LLMs-from-scratch)  
- [YouTube Playlist](https://www.youtube.com/watch?v=4yNswvhPWCQ&list=PLTKMiZHVd_2Licpov-ZK24j6oUnbhiPkm)  

This was hands-on and very clear, making it easier to understand concepts.  

---

### 10. Courses  
Some courses that really helped me understand concepts better:  
- [Attention in Transformers: Concepts and Code in PyTorch](https://learn.deeplearning.ai/courses/attention-in-transformers-concepts-and-code-in-pytorch/lesson/han2t/introduction)  
- [How Transformer LLMs Work](https://learn.deeplearning.ai/courses/how-transformer-llms-work/lesson/nfshb/introduction)  
- [Build LLM Apps with LangChain.js](https://learn.deeplearning.ai/courses/build-llm-apps-with-langchain-js/lesson/vchyb/introduction)  

---

### 11. Book  
I am reading the book **“Introduction to Machine Learning” by Ethem Alpaydin**.  
Whenever I am not clear about a concept, I refer to this book to strengthen my understanding.  

- [Book Link](https://www.google.co.in/books/edition/Introduction_to_Machine_Learning/1k0_-WroiqEC?hl=en&gbpv=0)  

---

### 12. Hands-On Reinforcement Learning  
I watched this video to get introduced to **Reinforcement Learning (RL)**:  
- [Introduction to RL](https://www.youtube.com/watch?v=wz141j9qIaU)  

---

### 13. Project: Minimal Lunar Lander – DQN  
I built this **Reinforcement Learning project** using **Gymnasium** and **Stable Baselines3**.  

- **GitHub link:** [Minimal Lunar Lander DQN](https://github.com/thinley4/Minimal-Lunar-Lander-DQN)  

---

From today, I will document my daily learning progress.

**Day 1: Q-Learning**

**Exploration vs. Exploitation**  
- Balancing exploration (trying new actions) and exploitation (choosing known best actions) is crucial in reinforcement learning.  
- Q-Learning is a technique that helps agents learn optimal actions through experience.

![One](https://github.com/thinley4/Learning_Artificial_Intelligence/blob/main/images/day1/one.png)

**Epsilon-Greedy Strategy**  
- The agent starts with a high exploration rate (E = 1), taking random actions to discover new possibilities.  
- As learning progresses, E decreases, and the agent exploits its knowledge more by choosing actions with higher Q-values.

![two](https://github.com/thinley4/Learning_Artificial_Intelligence/blob/main/images/day1/two.png)

**How Actions Are Chosen**  
- At the initial state, the agent selects actions randomly due to high exploration.  
- Over time, the agent uses the epsilon-greedy strategy to balance exploration and exploitation.  
- When exploiting, the agent picks the action with the highest Q-value for the current state from the Q-table.

![three](https://github.com/thinley4/Learning_Artificial_Intelligence/blob/main/images/day1/three.png)
![four](https://github.com/thinley4/Learning_Artificial_Intelligence/blob/main/images/day1/four.png)

**Q-Learning Process**  
- After each action, the agent observes the next state and reward, then updates the Q-value in the Q-table for the previous state-action pair.

**Resource:**  
- [Q-Learning Video](https://deeplizard.com/learn/video/mo96Nqlo1L8)


**Day 2: Markov Decision Processes (MDPs)**

A Markov Decision Process models the sequential decision-making of an agent interacting with an environment. At each step, the agent selects an action from the current state, transitions to a new state, and receives a reward. This sequence of states, actions, and rewards forms a trajectory.

The agent’s objective is to maximize the cumulative reward over time, not just the immediate reward from each action. This encourages the agent to consider long-term benefits when making decisions.

**MDP Mathematical Representation**

![one](https://github.com/thinley4/Learning_Artificial_Intelligence/blob/main/images/day2/one.png)
