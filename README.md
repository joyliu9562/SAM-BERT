This is the official implementation of SAM-BERT

Here we provide:

* Source code for SAM-BERT in ``model.py``
* Dependency utils in ``utils.py``
* Data for training and testing in ``/data/``
* Use ``data_loader.py`` to load data
* For training and testing use `train_test.py` 

## Abstract:

Life trajectories of notable people, consisting of (person, time, location, activity type) tuples, record times and locations of their births, deaths, education, jobs, and marriages, as well as when and where they won an election, made a scientific discovery, finished a masterpiece, came up with an invention, became a champion, and ended a war. These undoubtedly convey essential messages for the wide work on human dynamics. However, current studies are limited to certain types of activities such as births and deaths which are easier to collect and extract -- there lack large-scale trajectories covering fine-grained activity types. Adopting a tool that extracts (person, time, location) triples from Wikipedia, we formulate a problem of classifying the triples into 24 carefully defined types, given the triples' textual context from Wikipedia. Instead of classifying mere text, we actually classify triples, and the semantic relations between the triple entities and their related text should be well emphasized and aggregated. Apart from the difficulty in multi-classification, it is challenging since the triple entities are often scattered afar in the context, with plenty of noise around. We make use of the syntactic graphs that bring the triple entities and their relevant information closer, and fuse them with text embeddings. The overall syntactic graph embeddings and the embeddings of the triple sub-graph are obtained through graph attention, and concatenated as the representation of the context and the triple. Specifically, the overall syntactic graph embeddings are learned by initiating the graph with BERT embeddings guided through its MASK to focus on the local triple, while the embeddings of the local triple sub-graph are initiated with general BERT embeddings. Forming this overall-local cross, text representations are well blended with overall and local awareness of the syntactic graph. Furthermore, we use Large Language Model (LLM) to reconstruct crowd-sourced Wikipedia text such that the extracted syntactic graphs are refined into more standard and unified formalities. Our method achieves an Accuracy of 83.13\%, surpassing the baselines. We showcase how fine-grained life trajectories can support grand narratives of human dynamics across time and space, by analyzing trajectories of 58,844 people. Besides, we make the code and the manually-labeled dataset publicly available.

## Requirements

See requirements.txt

