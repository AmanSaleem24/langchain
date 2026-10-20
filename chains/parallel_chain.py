from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_classic.schema.runnable import RunnableParallel

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

prompt1 = PromptTemplate(
    template = "Generate a short and simple notes from following text \n {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template = "Generate a 10 question/answers from the following text\n {text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template = "Merge the provided notes and quiz into a single document\n notes -> {notes} and \n quiz -> {quiz}",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

paraller_chain = RunnableParallel({
    "notes": prompt1 | model | parser,
    "quiz" : prompt2 | model | parser
})

merge_chain = prompt3 | model | parser

chain = paraller_chain | merge_chain

text = """
# Artificial Intelligence, Machine Learning, and Deep Learning

Artificial Intelligence (AI) is a branch of computer science that focuses on developing systems capable of performing tasks that typically require human intelligence. These tasks include reasoning, problem-solving, understanding natural language, recognizing images, making decisions, and learning from experience. AI is used in many applications, including virtual assistants, recommendation systems, autonomous vehicles, medical diagnosis, fraud detection, and robotics.

AI can be divided into different categories based on its capabilities and approach. Narrow AI, also called weak AI, is designed to perform specific tasks. Examples include voice assistants, spam filters, and movie recommendation systems. These systems can perform their assigned tasks effectively but cannot automatically generalize their abilities to every domain. Artificial General Intelligence (AGI), on the other hand, refers to a hypothetical form of AI that could learn, reason, and perform a wide variety of intellectual tasks at a level comparable to humans. Current AI systems have not achieved AGI.

Machine Learning (ML) is a subset of AI that enables computers to learn patterns from data rather than relying exclusively on explicitly programmed rules. A traditional program follows instructions written by a developer, whereas a machine learning model learns relationships from examples during training. After training, the model uses the patterns it has learned to make predictions or decisions about new data.

Machine learning is commonly divided into three major categories: supervised learning, unsupervised learning, and reinforcement learning.

Supervised learning uses labeled training data, where each example contains input features and a known target output. The model learns a mapping between inputs and outputs. Classification and regression are two common supervised learning tasks. Classification predicts discrete categories, such as determining whether an email is spam or legitimate. Regression predicts continuous numerical values, such as estimating the price of a house based on its size, location, and other characteristics.

Unsupervised learning works with data that does not have predefined target labels. The objective is to discover hidden patterns, structures, or relationships within the data. Clustering is a common unsupervised learning technique that groups similar data points together. For example, a retailer might use clustering to identify customer segments based on purchasing behavior. Dimensionality reduction is another technique that represents data using fewer variables while attempting to preserve important information. It can help with visualization, noise reduction, and computational efficiency.

Reinforcement learning involves an agent that interacts with an environment and learns through rewards and penalties. At each step, the agent observes the current state, selects an action, and receives feedback from the environment. The goal is to learn a policy that maximizes cumulative reward over time. Reinforcement learning is used in areas such as robotics, game-playing systems, and resource allocation. Unlike supervised learning, it does not necessarily require a labeled correct answer for every action.

Deep Learning is a specialized subset of machine learning that uses artificial neural networks with multiple layers to learn increasingly complex representations of data. These networks are loosely inspired by the organization of biological neurons, although artificial neurons operate through mathematical computations rather than biological processes. Deep learning has achieved significant results in image recognition, speech processing, natural language processing, and generative AI.

An artificial neural network typically consists of an input layer, one or more hidden layers, and an output layer. The input layer receives the features of the data. Hidden layers transform those features using weighted connections, biases, and activation functions. The output layer produces the model's prediction. During training, the network adjusts its parameters to reduce the difference between its predictions and the desired outputs.

A common training method for neural networks is backpropagation, which calculates how changes in network parameters affect the loss function. An optimization algorithm, such as stochastic gradient descent or Adam, uses gradient information to update the parameters. The learning rate controls the size of these updates. If the learning rate is too high, training may become unstable; if it is too low, training may progress very slowly.

The loss function measures how far a model's predictions are from the desired outcomes. For example, cross-entropy loss is commonly used for classification, while mean squared error is frequently used for regression. The training process attempts to minimize this loss, but a low training loss does not necessarily mean that the model will perform well on new data.

To evaluate generalization, datasets are commonly divided into training, validation, and test sets. The training set is used to learn model parameters. The validation set helps select hyperparameters and compare model configurations. The test set provides a final evaluation on data that was not used for training or model selection. Repeatedly tuning a model against the test set can compromise the reliability of the final evaluation.

Two important problems in machine learning are overfitting and underfitting. Overfitting occurs when a model learns the training examples too closely, including noise or accidental patterns, and consequently performs poorly on unseen data. Underfitting occurs when a model is too simple or insufficiently trained to capture the underlying relationships. Techniques such as regularization, dropout, data augmentation, and early stopping can help reduce overfitting. Choosing an appropriate model architecture and training procedure can help address both problems.

Natural Language Processing (NLP) is an area of AI concerned with enabling computers to process and generate human language. Common NLP tasks include sentiment analysis, machine translation, text classification, summarization, and question answering. Earlier NLP systems often relied heavily on manually designed linguistic features and statistical methods. Modern systems increasingly use neural networks and large language models.

Transformers are a neural network architecture that has become central to modern NLP. A key component is the attention mechanism, which allows a model to determine how strongly different tokens should influence the representation of a particular token. Self-attention uses query, key, and value vectors to calculate relationships among tokens. This enables the model to capture contextual dependencies across a sequence more effectively than many earlier architectures.

Large Language Models (LLMs) are trained on large text datasets to learn statistical patterns in language. Many are trained initially through next-token prediction, in which the model learns to predict the next token given the preceding context. During inference, the model generates text one token at a time, using a decoding strategy to select each next token. Some models are further adapted using supervised fine-tuning, preference optimization, or other techniques to improve their usefulness and behavior.

Generative AI refers to systems that create new content, such as text, images, audio, video, or code. Large language models are one category of generative AI. These systems can assist with writing, programming, tutoring, research, and information retrieval. However, they can also generate inaccurate statements, fabricated references, or plausible-sounding explanations that are not supported by evidence. Their outputs should therefore be evaluated, particularly in high-stakes settings.

Retrieval-Augmented Generation (RAG) combines information retrieval with text generation. A typical RAG system first converts documents into smaller chunks and represents them as numerical embeddings. These embeddings are stored in a vector database or another retrieval index. When a user asks a question, the system retrieves relevant chunks and supplies them as context to a language model. The model then generates a response grounded in the retrieved information. RAG can improve access to domain-specific or frequently updated information, although retrieval quality and the model's handling of context remain important limitations.

AI systems also raise ethical and practical concerns. Training data may contain biases that lead to unfair outcomes. Models can expose sensitive information, consume substantial computational resources, or make decisions that are difficult to explain. Responsible AI development involves evaluating accuracy, fairness, privacy, robustness, transparency, and the potential consequences of deployment. Human oversight is particularly important when AI influences healthcare, employment, finance, education, or other consequential decisions.

In conclusion, artificial intelligence is a broad field that includes machine learning, deep learning, natural language processing, and generative systems. Machine learning enables systems to discover patterns from data, while deep learning uses multilayer neural networks to learn complex representations. Transformers and large language models have expanded the capabilities of language-based AI, and techniques such as RAG help connect these models to external knowledge. Building reliable AI applications requires more than selecting a powerful model: it also requires appropriate data, careful evaluation, effective engineering, and responsible deployment.
"""

result = chain.invoke({"text": text})

print(result)

chain.get_graph().print_ascii()

