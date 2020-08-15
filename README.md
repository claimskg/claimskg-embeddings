# Can Multimodal Embeddings Tell Us What Fact-checked Claims Are About?
This repository contains all the code and data to reproduce the experiments in the paper entitled "Can Multimodal Embeddings Tell Us What Fact-checked Claims Are About?" on the basis of the ClaimsKG knowledge graph of fact-checked claims. 

Please first consult the details regarding the generation/preparation of the graph embeddings are detailed in the dedicated repository: [https://anonymous.4open.science/r/78d23c68-9302-41ad-88d7-b07a1ac9d975/.](https://anonymous.4open.science/r/11c0a5c3-bf88-4fa2-bac3-a777ae9e7d37/)
Please make sure to set-up the sparql endpoint as described using the virtuoso docker container. Please make sure to publish the port on the host machine. 

Further details about the dataset can be found here: [https://anonymous.4open.science/r/11c0a5c3-bf88-4fa2-bac3-a777ae9e7d37/](https://anonymous.4open.science/r/11c0a5c3-bf88-4fa2-bac3-a777ae9e7d37/).

You can reproduce the experiments with the following command. Make sure to replace the dataset and model paths by 
the right values on your system. 

```bash
python -m neg_claim_topic_classification "http://localhost:8890/sparql" data/gold_updated.csv /path/to/kbc/datasets /path/to/kbc/models
```

This command loads ALL models at once (several graph embedding models, DistilRoberta, GPT2) and requires at least 64GB
of RAM to run properly. If you have less, you may need to run the experiments one by one by commenting out 
appropriate sections of the scripts. No GPU is required but if you have one it will speed up the computation of the text 
embeddings. 












