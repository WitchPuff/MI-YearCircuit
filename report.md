# Finding Year Comparison Circuit in LLMs

> Yutong He
>
> yutong.he@stud.uni-heidelberg.de
>
> Mechanistic Interpretability 24/25 WS
>
> May 3rd, 2025



## 1 Research question

How does a medium-sized LLM internally decide whether one 4-digit year is later than another? Is it able to find out the former digits weights more than the latter ones when comparing temporal relations between two years?

This project focus on finding the mechanistic circuit, from the first token embedding, through attention heads and residual streams, to the final answer logit, using methods including probing and activation patching.





## 2 Data

### Prompt Design

The model is asked to answer if one year is earlier than the other, and the labels are binary as 'Yes'/'No'.

```text
"<system> You are a bot that can compare years. <system>
Please answer: Is 1933 earlier than 1642? Answer with 'Yes' or 'No'.
A:"
```

### Model Selection

Phi-3-mini-4k-instruct (≈ 3.8 B params) can reach a zero-shot accuracy around 95% on this task.

Importantly, this model tokenises each 4-digit year into 5 tokens, which looks like this: `1937` $\rightarrow$ `['_', '1', '9', '3', '7']`. This is convenient for the digit-wise analysis we need later.

The dataset is randomly generated with years ranging from 1000 to 2020. Among 150 samples, only the prompt questions that the model can correctly answer are included, resulting in a total of 134 samples.

For further ablation studies, an additional dataset with years ranging from 1800 to 2020 is constructed to observe the impact of digit frequency at each position.



## **3 Experiments & Results**

### Probing

For this experiment, I am trying to find out how the model encodes every digit in a 4-digit year. I apply probing approaches to observe whether there exist linear structures for digit-wise token embeddings among all layers, as well as whether the patterns alter between the two years appeared in a prompt.

#### 1) Methodology

1. Train a simple Ridge `reg` to see if it can fit the embedding of each layer after the residual modules to the corresponding years, and visualize the $R^2$ score of `reg` over layers and different digit positions or at the average level overall.
2. Train a simple linear regression `lr` to fit projection (`embedding @ normalized(reg.coef)` as time direction of this dimension) to the corresponding years, and visualize the projected years and actual years.
3. A simple ablation for the number ranges for each token position: 1800-2020 vs 1000-2020

Relative codes are implemented in `probing.py`.

#### 2) Results

Note: Token position 0 refers to the thousands digit, while token position 3 refers to the ones digit.

| $R^2$ Year Range: 1000-2020                                  | $R^2$ Year Range: 1800-2020                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![ridge_Year1](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/04/30/2d15d0a2773be30b60d379cf1a72c19b-ridge_Year1-d5728f.png) | ![ridge_Year1](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/04/30/118f3d6559e5c18673499d261b2d0bc3-ridge_Year1-1275b0.png) |
| ![ridge_Year2](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/04/30/3cef2d7e8a85468154b0ec6cbc1d6f7a-ridge_Year2-f3d175.png) | ![ridge_Year2](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/04/30/b1b09f39ac7bac0bb66d8fb7c413f120-ridge_Year2-f12b89.png) |



| Time Direction Year Range: 1000-2020                         | Time Direction Year Range: 1800-2020                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/fb63c1a5a1bb54ab05567be157077026-vtime_Residual_layer_0_Year1-8b9ade.png" alt="vtime_Residual_layer_0_Year1"  />![vtime_Residual_layer_0_Year2](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/bac992b37d600ce1b6fb8b7e4a965889-vtime_Residual_layer_0_Year2-5967d6.png) | <img src="https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/50de2c51165bfcb58114594ecb6984aa-vtime_Residual_layer_0_Year1-3af4bc.png" alt="vtime_Residual_layer_0_Year1"  />![vtime_Residual_layer_0_Year2](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/a0d90d9f01976f34e74de4861a631d9d-vtime_Residual_layer_0_Year2-620836.png) |
| ![vtime_Residual_layer_1_Year1](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/6f88d8b063d74731a07c46bba9182ae0-vtime_Residual_layer_1_Year1-01c510.png)![vtime_Residual_layer_1_Year2](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/1d2cebf0abdc8588a61bf67caf5bfa1f-vtime_Residual_layer_1_Year2-08bf44.png) | ![vtime_Residual_layer_1_Year1](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/468e6968f24d67ae66c40df39a6045fd-vtime_Residual_layer_1_Year1-132aa1.png)![vtime_Residual_layer_1_Year2](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/25c25d692b3ea247e37b0fae28120556-vtime_Residual_layer_1_Year2-4ee9c7.png) |
| ![vtime_Residual_layer_2_Year1](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/d8ccdd170bac1cc6104a72de841665bc-vtime_Residual_layer_2_Year1-e020f6.png)![vtime_Residual_layer_2_Year2](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/c5290c2c83af1366f7d03e91db9c76bb-vtime_Residual_layer_2_Year2-76c096.png) | ![vtime_Residual_layer_2_Year1](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/0f361f64e9f2b31e1ad62ed5ad098964-vtime_Residual_layer_2_Year1-fdbb0b.png)![vtime_Residual_layer_2_Year2](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/de022850f86d03a9493d10ab3af88464-vtime_Residual_layer_2_Year2-d47505.png) |
| ![vtime_Residual_layer_3_Year1](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/dfd0ff4d0984ab3550f578cf6416e5ca-vtime_Residual_layer_3_Year1-f5cf2e.png)![vtime_Residual_layer_3_Year2](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/e83c5175fdb278c461ee87773a20dd20-vtime_Residual_layer_3_Year2-a2ec29.png) | ![vtime_Residual_layer_3_Year1](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/976be07b0c249206096e3f4dc1fb5e2f-vtime_Residual_layer_3_Year1-8f0116.png)![vtime_Residual_layer_3_Year2](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/86ed43ab1f67d826600d2b104f6cd7af-vtime_Residual_layer_3_Year2-f00e3f.png) |
| ![vtime_Residual_layer_4_Year1](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/7b99351404873cf51a5bb5f1738c6f81-vtime_Residual_layer_4_Year1-a89f69.png)![vtime_Residual_layer_4_Year2](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/e82f5f98f71f2aa2560f11915855d09f-vtime_Residual_layer_4_Year2-3d55d8.png) | ![vtime_Residual_layer_4_Year1](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/3a48064cc24403d41e9c03a4caa244dd-vtime_Residual_layer_4_Year1-3540df.png)![vtime_Residual_layer_4_Year2](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/12f8d742cbc6584fc94c3bd58ce2b0dd-vtime_Residual_layer_4_Year2-2a8ef1.png) |
| ![vtime_Residual_layer_5_Year1](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/1bd9e631a11d0f9da52b494e31f633aa-vtime_Residual_layer_5_Year1-871d85.png)![vtime_Residual_layer_5_Year2](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/c27a35bc204a30fff67a5ddf9e27fd68-vtime_Residual_layer_5_Year2-e01da4.png) | ![vtime_Residual_layer_5_Year1](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/f1acac81a1e8061a5fa0d557c1cbc5c3-vtime_Residual_layer_5_Year1-d24d74.png)![vtime_Residual_layer_5_Year2](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/02f48ad3bc9c25c4169bf6629565d4d0-vtime_Residual_layer_5_Year2-02bce9.png) |
| ![vtime_Residual_layer_11_Year1](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/8ef38f9156b39234b2336ddb462248a1-vtime_Residual_layer_11_Year1-4b49b3.png)![vtime_Residual_layer_11_Year2](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/9407edf23bdd3258d5b39474d6effdc1-vtime_Residual_layer_11_Year2-802f08.png) | ![vtime_Residual_layer_11_Year1](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/23e33c2e500fb6b780631761d198e886-vtime_Residual_layer_11_Year1-fb3d11.png)![vtime_Residual_layer_11_Year2](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/dd4d80488a94b8351b21e08175466ef1-vtime_Residual_layer_11_Year2-6058d0.png) |
| ![vtime_Residual_layer_25_Year1](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/077a734ae091850ce698234531a9e47a-vtime_Residual_layer_25_Year1-d59262.png)![vtime_Residual_layer_25_Year2](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/cc31fe57bb19c6025ff74804bf72bcb4-vtime_Residual_layer_25_Year2-555a12.png) | ![vtime_Residual_layer_25_Year1](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/5067a513bfa03087b4ef25edb2e08799-vtime_Residual_layer_25_Year1-42c804.png)![vtime_Residual_layer_25_Year2](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/83cc383bc31224dbfa5b919faeecde46-vtime_Residual_layer_25_Year2-2c6e86.png) |
| ![vtime_Residual_layer_32_Year1](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/0350f57e7b42dd54090616b35c976a6b-vtime_Residual_layer_32_Year1-fb2153.png)![vtime_Residual_layer_32_Year2](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/f8e24550f75b0bf5cbdeb940b959747a-vtime_Residual_layer_32_Year2-264d1a.png) | ![vtime_Residual_layer_32_Year1](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/7c15d6d5e5fe77952a9819375ad2158e-vtime_Residual_layer_32_Year1-363c03.png)![vtime_Residual_layer_32_Year2](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/03/910b7472a9884c5a20fa082d9514cecd-vtime_Residual_layer_32_Year2-4a053a.png) |



#### 3) Conclusion

1. It fits better for year 2 (the one occurs later in the sentence), maybe it's because it is closer the answer position.
2. The deeper the layers, the higher the $R^2$, and it reaches 1 in first few layers (Layer 0-4). 
3. The deeper the layers, the better `lr` fits except for token at first digit. 
4. *Ablation*: If the range for the second digit scale up, it fits better in later layers than the last two digits, the $R^2$ is higher at the same time. Therefore, the emergence of this linear structure is correlative to the frequency and diversity of numbers appearing at every digit.





### Comparison Circuit

In this experiment, I aim to identify the comparison circuit in the LLM that captures relations digit-wise in the years.

#### 1) Methodology

1. With transformer-lens API, the model is run with cache. For each sample, the attention patterns, head outputs, and post-residual activations are stored and analysed at every layer, every head, and every digit level.

2. Three metrics are observed:

   1. `score_b2a`: Cross-attention score between same digits from a pair of years. `score_b2a = Attn(year₂_digit → year₁_digit)`. It measures how much the head explicitly looks from the second year’s digit to the matching digit of the first year.

      ```python
      def get_attn_score(cache, pos1, pos2, layer_id, head_id):
          pattern = cache[f"blocks.{layer_id}.attn.hook_pattern"][0, head_id]  # [Q, K]
          if pos1 > pos2:
              pos1, pos2 = pos2, pos1
          # score_a2b = pattern[pos1, pos2].item() # the previous a can't look at the latter b for causal modelling.
          score_b2a = pattern[pos2, pos1].item()
          return score_b2a
      ```

   2. `qk_diff`: It computes the cosine similarity between the QK scores of two digit positions from different years, with one negated. A high similarity implies that the attention pattern from one digit is roughly the inverse of the other, suggesting the head may be comparing the two digits via subtractive mechanisms.

      ```python
      def get_qk_diff(cache, pos1, pos2, layer_id, head_id):
          q = cache[f"blocks.{layer_id}.attn.hook_q"][0, :, head_id]
          k = cache[f"blocks.{layer_id}.attn.hook_k"][0, :, head_id]
      
          qk = q @ k.T
      
          logits_a = qk[pos1]
          logits_b = qk[pos2]
      
          sim = F.cosine_similarity(logits_a, -logits_b, dim=0).item()
      
          return sim
      
      ```

   3. `delta`: Using the TransformerLens API, the model is run with a hook that swaps the activations at specific digit token positions between two years, in order to observe the causal impact of the heads.

      Specifically, `delta` is computed as: 

      ```python
      def get_metric(logits):
          last_logits = logits[:, -1, :]
          t = model.to_single_token(true_label)
          f = model.to_single_token(false_label)
          return last_logits[:, t] - last_logits[:, f]
      baseline_score = get_metric(baseline_logits)
      patched_score = get_metric(patched_logits)
      delta = baseline_score - patched_score
      ```

Relative codes are implemented in `comepare_head.py`.

#### 2) Results

##### **Causality Analysis**

The causal influence of individual digit positions is captured by the difference `delta` in output probabilities of true labels between the original logits and those obtained by patching the corresponding digits in both years with zero.

To observe the importance of first digit, I constructed an ablation dataset by randomly sampling data points that differ in the first digit, since such changes are relatively infrequent in most cases. The size of datasets are both set to 10 samples here.

The layer- and head-wise heat maps of delta scores for each digit are shown below.

Note: Digit 0 refers to the thousands digit, while digit 3 refers to the ones digit.

| Delta   | Random samples<br />(only 2/10 samples different  in digit 0) | Samples that all differ in digit 0                           | Findings                                                     |
| ------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Digit 0 | ![image-20250502174944555](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/02/857c9925b61358fa356387fc5dee79ff-image-20250502174944555-f9b361.png) | ![image-20250502175701237](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/02/878273b99b72f2350c06ee4a32e9a0b6-image-20250502175701237-4d66ad.png) | 1. When using samples that all differ in the first digit, the overall `delta` scores of most heads for digit 0 increase noticeably. Causally significant heads are primarily concentrated in layers 0–14.<br /><br />2. Key heads consistently identified in both settings include: (10, 29), (7, 23);<br /><br />3. In contrast, heads such as (1, 30) and (2, 3) show divergent patterns between the two sample sets. This may indicate their involvement in modeling transiftions of the first digit in year representations. |
| Digit 1 | ![image-20250502174955184](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/02/1ca15e6341ee0924806e3b573eb4d085-image-20250502174955184-2ca6c4.png) | ![image-20250502175710386](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/02/ef0761c77a95e0f827c4490be1f852a7-image-20250502175710386-df3b8e.png) | Key head: (1, 9)                                             |
| Digit 2 | ![image-20250502175003385](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/02/91060aca19952f869467175074b84606-image-20250502175003385-c877c4.png) | ![image-20250502175717878](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/02/aaf0e5fce33c232b5b54ec42ded1f6d7-image-20250502175717878-4ef62e.png) | 1. When using samples that all differ in the first digit, the overall `delta` scores of most heads for digit 2 decrease, noticing the ranges shown by cbar. Causally significant heads are primarily concentrated in layers 0–14.<br /><br />2. One key head may be (1, 9), but it shows contrary pattern for two settings. |
| Digit 3 | ![image-20250502175010841](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/02/ffc1135640b241b2ad9d37157a401456-image-20250502175010841-5ca848.png) | ![image-20250502175739544](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/02/6a8401a9d4547aa335ad1f6f045403ee-image-20250502175739544-d668dd.png) | Key heads primarily emerge in layers 10-14, including (12, 22), (12, 30), (10, 2). |







##### **Attention Analysis**

I aim to find out if there are compare heads for different digits when completing the task of understanding the temporal relation of years. First of all, I look into the attention scores `score_b2a` from the second year’s digit to the corresponding digit of the first year. Moreover, I compute the QK difference `qk_diff` to testify if there are comparison behaviours by subtraction operations.

| Score_b2a | Random samples                                               | Samples that differ in digit 0                               | Findings                                                     |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Digit 0   | ![image-20250502175049342](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/02/9b489bc422ffa9ab76b190c3bcd6c0a2-image-20250502175049342-45e48d.png) | ![image-20250502175803068](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/02/25790698bd228a26054ea8766ef3603b-image-20250502175803068-210055.png) | 1. Layers 3–4 primarily capture relations between the first digits, indicating that the model attends to the most decisive digits early in the processing.<br /><br />2. Key heads: (10, 11), (9, 27), (1, 4), (1, 14) |
| Digit 1   | ![image-20250502175056357](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/02/e457e9608d553d6482d6eeb941324add-image-20250502175056357-0b8d22.png) | ![image-20250502175809918](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/02/49ae008bb545606a32d89398924d9180-image-20250502175809918-44cbe5.png) | Key head: (9, 27), (10,14), (10, 11), (11, 5)                |
| Digit 2   | ![image-20250502175326275](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/02/4501d5a2aae46ae76d547a1e44b07a51-image-20250502175326275-1cd2a7.png) | ![image-20250502175815885](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/02/567608177a469ede46cc5711512e2b86-image-20250502175815885-329697.png) | Key head: (9, 27), (10, 13), (10, 11), (4, 14)               |
| Digit 3   | ![image-20250502175334269](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/02/87736eefbb9786a05f45bb56e798b5c9-image-20250502175334269-bbe6eb.png) | ![image-20250502175832845](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/02/8f9000054e2e7c007f69d698eb2d8d47-image-20250502175832845-c78f0e.png) | Key head: (9, 27), (10, 14)                                  |

| qk_diff | Random samples                                               | Samples that differ in digit 0                               | Findings                                                     |
| ------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Digit 0 | ![image-20250502175610828](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/02/418a71aa900ffe9d3d598b4bf13a15a1-image-20250502175610828-982c1d.png) | ![image-20250502175844258](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/02/79a528d79e5dff85f16c9618122aed7f-image-20250502175844258-800d9f.png) | 1. Layers 2-8 have comparison behaviours based on subtraction operations.<br /><br />2. Key heads: (4, 3), (3, 11), (5, 24), (7, 2), (6, 7)<br /><br />3. For samples that differ in digit 0, there seems to be a "checking" stage at the last few deep layers, including heads (30, 24), (31, 11), (31, 21), (27, 11). |
| Digit 1 | ![image-20250502175617635](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/02/eb1f17d98c959608a0f796eb0694ae16-image-20250502175617635-49a554.png) | ![image-20250502175850360](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/02/c36262b5cedb89010f72a5215698ec00-image-20250502175850360-b3c644.png) | Key heads: (9, 3), (0, 4), (0, 18), (0, 15)                  |
| Digit 2 | ![image-20250502175624694](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/02/4b22b3a7e272edea77b80c892d7559fd-image-20250502175624694-7ecd4b.png) | ![image-20250502175855897](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/02/c290106388867f785385d8dbdf06e7cd-image-20250502175855897-6ee6d0.png) | Key heads: (0, 4), (0, 15), (0, 18), (16, 17)                |
| Digit 3 | ![image-20250502175632785](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/02/490c3b9eb7bdf1e9954c081d88f5af85-image-20250502175632785-f87ba5.png) | ![image-20250502175902431](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/02/9f7d8647abcddb44190d1bd716dd5334-image-20250502175902431-efbf60.png) | Key heads: (24, 9), (11, 13)                                 |

#### 3) Conclusion

In conclusion, the model seems to use compare heads to capture the relations between the first digit in the early stage. For the lower digits (hundreds, tens, ones), there is no clear layer that focuses on comparing same digits. Instead, a few mid-layer heads like (9,27), (10,13), and (10,14) consistently contribute. This suggests the model compares the first digit (thousands) early and explicitly via compare heads, while encoding the other digits more diffusely—possibly linearly. This matches the probing results, where lower digits show linear structure in early layers, while the first digit does not. 

Additionally, the `qk_diff` analysis also shows strong difference-based comparison only for the first digit, mostly in layers 2-8, which is mostly absent for the other digits. 

However, the attention analysis results do not fully align with the causality analysis, suggesting that the decision circuit is long and complex. For example, compare heads may compute differences early, but the final decision could depend on later heads that gather and process this information through the residual stream.



### Answer Circuit

For this experiment, the goal is to find out the answer circuit in the model, using methods of activation patching. 

#### 1) Methodology

1. Using the transformer-lens API, the model is run on a set of yes/no classification prompts. The attention head outputs are cached.

2. Focusing on the answer token position (`answer_pos = tokens.shape[-1] - 1`), for every layer and attention head, three metrics are computed:

   1. `delta`: It measures the drop in logit margin between the true and false labels when the token position of true label is patched to zero. A higher drop indicates greater causal importance, similar to how `delta` is used in identifying comparison circuits.

   2. `cos_true`: It computes the cosine similarity between a head’s output at the answer token and the unembedding vector of the true label. Higher values suggest the head helps copy and write the correct answer to the final decision.

      ```python
      head_out = base_cache[f"blocks.{layer_id}.attn.hook_result"][0, answer_pos, head_id]
      true_label_dir  = model.W_U[:, model.to_single_token(true_label)]
      cos_true  = F.cosine_similarity(head_out, true_label_dir, dim=0).item()
      ```

   3. `Top5 attended tokens frequency`: It counts the frequency of top5 tokens that receive the highest attention from the answer token position, showing which parts of the input the model focuses on most when generating the answer.

      ```python
      attn = base_cache[f"blocks.{layer_id}.attn.hook_pattern"][0, head_id, answer_pos]
      top_idx = attn.topk(5).indices.tolist()
      top_tokens = [model.to_string(int(tokens[0, i])) for i in top_idx]
      ```

Relative codes are implemented in `answer_head.py`.

#### 2) Results

##### **Causality Analysis**

![image-20250502230119266](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/02/190afdd3ca568808009d8b57437cf46a-image-20250502230119266-4e1f10.png)

- Key heads that copy the true label token: (28, 10), (22, 30), (24, 26), (27, 3), (30, 24)
- Key heads that has significant causal importance: (16, 19), (18, 2), (14, 19), (6, 12), (9, 4)

##### **Top Attended Tokens**

![image-20250502232019062](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/2025/05/02/7bbf3bf8e7a1dc5238743d5051a8bcaa-image-20250502232019062-8f24da.png)





#### 3) Conclusion

The heatmap reveals that the answer token frequently attends to \<s> (the EOS token), and key semantic words like “Answer”, “Yes”, and “No”, especially in mid-to-late layers. Notably, several heads with high alignment to the true label token—such as (22, 30), and (24, 26)—also consistently direct attention to these answer-related tokens, indicating their role in writing the final output.













## 4 Conclusion & Future Work

The current analysis suggests that the model solves the “which year is later?” task using a multi-stage pipeline:

In the embedding and early residual layers (0–4), the model learns nearly perfect linear representations for the hundreds, tens, and ones digits. However, the thousands digit remains nonlinear at this stage.

To handle the first digit, the model uses a distinct comparison circuit. Layers 3–4 primarily capture attention relationships between the most decisive first digits early in the processing. Correspondingly, QK-difference analysis supports the emergence of strong difference-based comparisons focusing exclusively on the first digit, particularly in layers 2–8. This may indicate that the model understands the first digit using comparison head mechanisms rather than the linear encoding for the lower digits, as shown in the probing experiments.

The comparison of the lower digits is more distributed. A small cluster of heads in the mid-layers, especially heads (9, 27) and (10, 11) along with their neighbor, contributes significantly to this process, though without being concentrated in any single layer.

Finally, in the later stages (layers 22-32), dedicated “copy heads” such as (22, 30) and (30, 24) focus on output tokens like “Answer,” “Yes,” “No,” or the end-of-sequence symbol `<s>`. These heads directly copy the appropriate token into the unembedding layer, finalizing the decision-making process.

However, some pieces of the jigsaw are still missing. Specifically, the exact residual pathways that carry the thousands-digit subtraction signal from the early comparison heads to the mid-layer integrators have not yet been traced. Additionally, the role and activation patterns of the relevant MLP neurons remain unclear. Furthermore, most heads with high attention scores or strong QK-difference signals do not consistently align with high causality scores, suggesting that the link between the “comparison circuit” and the final “decisive circuit” is still incomplete.

To close this gap, further work is needed. In particular, more fine-grained ablations would be usefu, such as prompts where only specific digits are altered, neuron-level analyses to probe the internal logic of the computation, and more diverse activation patching strategies to map the flow of information throughout the model. These approaches could help clarify how digit-level comparisons are integrated and ultimately translated into the model’s final decision.





