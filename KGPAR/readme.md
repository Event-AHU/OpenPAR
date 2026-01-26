# Knowledge Graph guided Cross-modality Hypergraph Learning for PAR 

## Abstract 
Current Pedestrian Attribute Recognition (PAR) algorithms typically focus on mapping visual features to semantic labels or attempt to enhance learning by fusing visual and attribute information. However, these methods fail to fully exploit attribute knowledge and contextual information for more accurate recognition. Although recent works have started to consider using attribute text as additional input to enhance the association between visual and semantic information, these methods are still in their infancy. To address the above challenges, this paper proposes the construction of a multi-modal knowledge graph, which is utilized to mine the relationships between local visual features and text, as well as the relationships between attributes and extensive visual context samples. Specifically, we propose an effective multi-modal knowledge graph construction method that fully considers the relationships among attributes and the relationships between attributes and vision tokens. To effectively model these relationships, this paper introduces a knowledge graph-guided cross-modal hypergraph learning framework to enhance the standard pedestrian attribute recognition framework. Comprehensive experiments on multiple PAR benchmark datasets have thoroughly demonstrated the effectiveness of our proposed knowledge graph for the PAR task, establishing a strong foundation for knowledge-guided pedestrian attribute recognition. 

## Framework  
![framework_new (1)](https://github.com/user-attachments/assets/1bab48be-e172-44ad-b94d-c86022487444)


## Configuration 


### Data Preparation
```
cd dataset/preprocess
python rap1_pad.py
cd dataset/global_hyergraph
python rap.py
```


## Training Script 
```python
python train.py RAPV1 --use_div --use_textprompt --use_vismask --use_GL --use_mm_former
```


