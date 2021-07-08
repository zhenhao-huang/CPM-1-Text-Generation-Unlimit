# CPM-1-Text-Generation-Unlimit
**CPM (Chinese Pretrained Models)** 是**北京智源人工智能研究院**和**清华大学研究团队**合作开展的大规模预训练模型。本项目为[**CPM-1-Finetune-Text-Generation**](https://github.com/zhenhao-huang/CPM-1-Finetune-Text-Generation)的**无限生成**代码，可用于**短文本生成**和**长文本生成**，具体任务对应于**歌词**和**小说**。[[**CPM官网**](https://cpm.baai.ac.cn/)][[**模型下载**](https://cpm.baai.ac.cn/download.html)][[**技术报告**](https://arxiv.org/abs/2012.00413)][[**CPM生成源码**](https://github.com/TsinghuaAI/CPM-1-Generate)]

更多流程细节参考[**博文**](https://blog.csdn.net/weixin_41611054/article/details/118522551)。
## 1 Model
微调完成后，以**CPM-Generate**模型为例，目录结构如下：

    .
    ├── 80000
    │   ├── mp_rank_00_model_states.pt
    │   └── mp_rank_01_model_states.pt
    └── latest_checkpointed_iteration.txt
## 2 Install
安装**基础依赖**：

    pip install -r requirements.txt
安装**apex**：

    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
安装**deepspeed**：

    pip install deepspeed
## 3 Use
每个任务的生成脚本有两种，一个是基于**CPM**[**最初的生成代码**](https://github.com/TsinghuaAI/CPM-1-Generate/commit/2422770187eae8d498292b16f46bb9dac71c3631#diff-159073d23d683b271e9842fac43ac6da5e70d95a69f32b6238d88639138d933a)改的无限生成脚本，另一个是基于加入`past_key_values`的**CPM生成代码**改的无限生成脚本。**无限生成**具体做法为使用**滑动窗口**，循环替换指定的`tokens`。

使用了`past_key_values`后，可以**提高生成速度**。当使用`past_key_values`无限生成时，超过`1024`生成的文本会**完全不通顺**(暂时没解决)，所以建议使用**最初的代码无限生成**。
### 短文本(歌词)
使用**动态滑动**窗口，打开`generate_lyric.py`、`generate_lyric_fast.py`中的相应注释即可。

运行命令：

    bash scripts/lyric/generate_lyric.sh /model_path example.txt
使用`past_key_values`运行命令：

    bash scripts/lyric/generate_lyric_fast.sh /model_path example.txt
`/model_path`为模型路径，`example.txt`为需要预测的输入文本，格式为：
    
    他们都懂得现实如同苦难戏<eod>
    他们都知道虚幻只出现在电视剧<eod>
    挥洒着金钱只是我的本性<eod>
    这就是为什么许多人都叫我mr King<eod>
### 长文本(小说)
运行命令：

    bash scripts/novel/generate_novel.sh /model_path example.txt
使用`past_key_values`运行命令：

    bash scripts/novel/generate_novel_fast.sh /model_path example.txt
`/model_path`为模型路径，`example.txt`为需要预测的输入文本，格式为：

    卡奇卡说："我知道一些：反物质是恐龙物理学家们猜想中的一种物质，它的原子中的粒子电荷与我们世界中的物质相反。反物质一旦与我们世界的正物质相接触，双方的质量就全部转化为能量。"乔耶点点触须说："现在大家知道有比核武器更厉害的东西了，在同样的质量下，正反物质湮灭产生的能量要比核弹大几千倍！"
可以通过调整生成脚本参数`out-seq-length`，控制**生成长度**。`past_key_values`脚本默认为`1024`。
### 单卡或多卡
默认的模型并行参数为`2`，如果需要修改，可以使用`change_mp.py`，并调整**生成脚本**中的`MPSIZE`。change_mp.py的使用示例如下：

    python change_mp.py /path/to/CPM MPSIZE
这里的`/path/to/CPM`为模型路径，`MPSIZE`为一个整数，可以为`1`或者`2`的倍数，结果会生成一个新的模型，存储路径为`/path/to/CPM_MPSIZE`。
## 4 Cite
    @article{cpm-v1,
      title={CPM: A Large-scale Generative Chinese Pre-trained Language Model},
      author={Zhang, Zhengyan and Han, Xu, and Zhou, Hao, and Ke, Pei, and Gu, Yuxian and Ye, Deming and Qin, Yujia and Su, Yusheng and Ji, Haozhe and Guan, Jian and Qi, Fanchao and Wang, Xiaozhi and Zheng, Yanan and Zeng, Guoyang and Cao, Huanqi and Chen, Shengqi and Li, Daixuan and Sun, Zhenbo and Liu, Zhiyuan and Huang, Minlie and Han, Wentao and Tang, Jie and Li, Juanzi and Sun, Maosong},
      year={2020}
    }
