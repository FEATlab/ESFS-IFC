# ESFS-IFC: Evolutionary streaming feature selection via incremental feature clustering

This repository implements ESFS-IFC as presented in the IEEE TEVC 2026 paper. ESFS-IFC is an evolutionary computation method combined with incremental feature clustering for feature selection with streaming data. 

<img width="1992" height="690" alt="image" src="https://github.com/user-attachments/assets/1e3f0c8d-b5cf-4c73-bcfe-a082a49baf75" />


# Abstract
In many real-world scenarios, features could arrive in groups over time, and the total size of the feature space is often unknown. Streaming feature selection is a commonly-used approach to addressing such dynamic scenarios, where newly arriving features must be assessed for both their relevance and redundancy with previously selected features. To effectively solve such a task, this paper proposes an evolutionary streaming feature selection method via incremental feature clustering that comprises three stages. First, the online irrelevant feature filtering stage eliminates irrelevant streaming features to reduce the noise effect and shrink the search space. Second, the incremental redundant feature clustering stage groups mutually redundant features into clusters, adaptively creating or merging feature clusters to adapt to the dynamically-changing feature sets. Finally, the interactive feature subset search stage identifies representative features within each cluster to form the best feature subset. Experimental results on 18 real-world datasets demonstrate that the proposed method has better classification performance than the compared state-of-the-art methods.

# Acknowledge
Please kindly cite this paper in your publications if it helps your research:
```
@article{zhang2026evolutionary,
title={Evolutionary streaming feature selection via incremental feature clustering},
author={Zhang, Yong and Jiao, Ruwang and Xue, Bing and Zhang, Mengjie},
journal={IEEE Transactions on Evolutionary Computation},
doi={10.1109/TEVC.2026.3667279},
year={2026},
publisher={IEEE}
}
```

# License
This code is released under the MIT License.

# Contact
If you face any difficulty with the implementation, please refer to: zyohhh@163.com

