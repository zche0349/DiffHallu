# DiffHallu

DiffHallu 是一个面向大模型长文本生成幻觉检测的、与具体模型无关的方法实现。它在同一对话前缀下调用同一目标模型生成多个候选续写，通过语义等价判断和顺序稳健的代表式聚类构造语义簇分布，最后计算平均加权语义熵，作为连续幻觉风险分数。

本仓库提供核心方法实现，配套实验与复现材料通过以下链接提供。

## 实验与复现材料

[Google Drive：DiffHallu 实验与复现材料](https://drive.google.com/drive/folders/12O1JY1aL4N-JBhYNvknP7U8ISY03HDMf?usp=sharing)

资料包括处理后的实验数据与固定划分、生成候选与真值标签、实验及基线脚本和配置、探针模型文件、运行环境与依赖版本说明，以及实验结果和复现工具。目录结构和使用方法见云盘中的 README。

## 方法流程

对于原始提示 `x`、模型已经生成的长文本 `z` 和目标实体 `e`，DiffHallu 执行以下步骤：

1. 在 `z` 中定位包含 `e` 的目标句并提取其左侧上下文。
2. 由 DiffHallu 构造完整的受控改写提示词，通过用户提供的 API 接口调用改写模型，将目标句改写为以 `e` 结尾的自然表达。
3. 将句末实体替换为 `[BLANK]`，构造固定生成前缀。
4. 调用用户提供的目标模型接口，在同一前缀下随机生成 `M` 个候选续写。
5. 截取每个候选的第一个完整句子，并取得 teacher-forcing 词元对数概率。
6. 调用用户提供的语义模型接口，进行双向 NLI 判断，构造语义等价矩阵。
7. 依次改变扫描起点执行代表式聚类，对每种聚类结果计算加权语义熵。
8. 对所有扫描顺序下的加权语义熵取平均，得到连续幻觉风险分数。

风险分数越高，表示候选续写的事实语义分布越分散，目标位置的幻觉风险越高。

## 安装

```bash
pip install -e .
```

项目要求 Python 3.10 或更高版本。运行时只依赖 `pysbd`，用于英文句子边界识别。

## 模型接口

DiffHallu 不选择、不配置、不加载也不部署任何具体模型。用户需要实现三个接口：

### `RewriterBackend`

构造前缀属于 DiffHallu 方法的一部分，因此完整的改写提示词、Few-shot 示例、JSON 解析、实体句尾检查和 `[BLANK]` 校验均由本仓库实现。用户只需要提供自己的 API 调用接口：接收 DiffHallu 构造的 prompt，并返回 API 的原始文本响应。

API 地址、鉴权方式、重试策略和改写模型均由用户自行配置，本仓库不保存任何 API 地址或密钥。

### `TargetModelBackend`

提供两个能力：

- `generate()`：在相同提示和前缀下生成指定数量的候选续写；
- `token_logprobs()`：使用 teacher-forcing 返回指定候选中每个词元的条件对数概率。

候选生成阶段要求用户自行部署目标模型。聊天模板、分词器、停止词元、设备、批处理和模型加载全部由用户的接口实现负责。目标模型除了生成候选外，还必须支持对指定候选进行 teacher-forcing，并返回逐词元条件对数概率；仅能返回文本的生成接口无法完整运行 DiffHallu。

## 默认对话模板与系统提示词

`diffhallu.templates` 只提供实验涉及的四个目标模型对应的参考对话模板和系统提示词：

- Qwen2.5，对应模板键 `qwen`，系统提示词为 `You are a helpful assistant.`；
- Meta-Llama-3.1，对应模板键 `llama-3`，系统提示词为 `You are a helpful assistant.`；
- Gemma-2，对应模板键 `gemma`，不额外添加独立 system 消息；
- Mistral-Small，对应模板键 `mistral`，不额外添加独立 system 消息。

用户的目标模型适配器可以直接调用：

```python
from diffhallu import COMMON_STOP_TOKENS, render_conversation_prompt

formatted_prompt = render_conversation_prompt(
    model_family="qwen",
    instruction=prompt,
    prefix=prefix,
)
```

`COMMON_STOP_TOKENS` 提供原实现使用的常见停止标记。如果所部署模型的 tokenizer 提供官方 `chat_template`，用户也可以优先使用模型自身模板，但应保持相同的用户指令和固定生成前缀。

### `SemanticBackend`

接收 `premise` 和 `hypothesis`，返回以下三个标签之一：

- `entailment`
- `neutral`
- `contradiction`

DiffHallu 会自动执行两个方向的判断，并固定采用论文中的宽松双向判定规则：任一方向为
`contradiction` 时不等价；两个方向均为 `neutral` 时不等价；其余情况
判为等价。本仓库不再提供其他判定模式。

接口可以基于 Transformers、vLLM、SGLang、远程 API 或内部推理平台实现。

## 论文参考复现模型

本仓库只保留模型接口，不包含任何具体模型实现或部署配置。若要对齐论文中
记载的模型选择：

- `RewriterBackend` 需要由使用者通过自己的 API 实现。论文有效性分析中的
  目标句调整辅助模型记载为 **Qwen3.5**；这是参考复现选择，不是 DiffHallu
  内置或强制配置的模型；
- `SemanticBackend` 的参考模型为 **DeBERTa-v2-xxlarge-MNLI**，其输出需映射为
  `entailment`、`neutral` 和 `contradiction`。

API 地址、密钥、模型路径、推理框架和设备均由使用者自行负责。

## 调用方式

```python
from diffhallu import DiffHallu


class MyRewriter:
    def complete(self, prompt, config):
        # 把 DiffHallu 构造好的提示词发送到自己的 API，返回原始文本。
        return call_my_api(
            prompt=prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )


class MyTargetModel:
    def generate(self, prompt, prefix, config):
        # 调用自己的目标模型，返回 config.num_samples 个候选。
        # 可以使用 diffhallu.templates 拼接系统提示词、用户提示和前缀。
        raise NotImplementedError

    def token_logprobs(self, prompt, prefix, completion):
        # 返回 completion 对应的 teacher-forcing 词元对数概率。
        raise NotImplementedError


class MySemanticModel:
    def predict(self, premise, hypothesis):
        # 返回 entailment、neutral 或 contradiction。
        raise NotImplementedError


detector = DiffHallu(
    rewriter=MyRewriter(),
    target_model=MyTargetModel(),
    semantic_model=MySemanticModel(),
)

result = detector.detect(
    prompt="Write a short biography.",
    generated_text="The report was written by Ada Lovelace.",
    target_entity="Ada Lovelace",
)

print(result.risk_score)
print(result.to_dict())
```

如果目标实体在长文本中出现多次，应传入 `occurrence_index`，或者直接传入准确的 `char_start` 和 `char_end`。

## 默认参数

| 参数 | 默认值 |
| --- | ---: |
| 前缀改写 API `max_tokens` | `768` |
| 前缀改写 API `temperature` | `0.7` |
| 候选续写数量 `num_samples` | `10` |
| 最大生成长度 `max_new_tokens` | `50` |
| 随机采样 `do_sample` | `True` |
| 温度 `temperature` | `1.0` |
| 核采样 `top_p` | `0.9` |
| `top_k` | `50` |

这些是 DiffHallu 的方法默认参数，不是模型部署配置。直接使用 `DiffHalluConfig()` 即采用表中的默认值。如需自定义，可以显式覆盖部分参数：

```python
from diffhallu import DiffHalluConfig, GenerationConfig

config = DiffHalluConfig(
    generation=GenerationConfig(
        num_samples=12,
        temperature=0.8,
        top_p=0.95,
        top_k=40,
    )
)
```

以下行为属于论文方法的固定步骤，不是 `DiffHalluConfig` 的可选参数：

- 使用上文所述的宽松双向 NLI 语义等价规则；
- 每个候选续写固定截取第一个完整句子；
- 对 `M` 个候选执行全部 `M` 种循环起点扫描，并对得到的 `M` 个加权语义熵取平均。

## 接口约束

`generate()` 必须使用同一个提示和前缀，并返回恰好 `num_samples` 个字符串。`token_logprobs()` 必须返回指定候选续写中所有词元的条件对数概率。DiffHallu 会计算这些概率的算术平均值，用于构造语义簇支持权重。

目标模型接口必须自行处理聊天模板、分词、停止词元、padding、设备、显存和批处理。DiffHallu 不对这些部署细节作任何假设。

## 数据集

DiffHallu 原始方法使用 Hugging Face 上的 [obalcells/longfact-annotations](https://huggingface.co/datasets/obalcells/longfact-annotations) 实体级标注数据。与论文四个目标模型对应的配置为：

- `Qwen2.5-7B-Instruct`
- `Meta-Llama-3.1-8B-Instruct`
- `gemma-2-9b-it`
- `Mistral-Small-24B-Instruct-2501`

该数据集目前还提供 `Llama-3.3-70B-Instruct` 配置，但它不属于论文评测的四个目标模型。

每条数据应按以下字段映射到 DiffHallu：

```python
row = dataset_row
prompt = row["conversation"][0]["content"]       # user 消息
generated_text = row["conversation"][1]["content"]

for annotation in row["annotations"]:
    target_entity = annotation["span"]
    char_start = annotation["index"]
    char_end = char_start + len(target_entity)

    result = detector.detect(
        prompt=prompt,
        generated_text=generated_text,
        target_entity=target_entity,
        char_start=char_start,
        char_end=char_end,
    )
```

其中 `annotation["index"]` 是实体在 assistant 消息中的字符起始偏移。本仓库不复制或重新发布数据，使用者应从 Hugging Face 数据集页面自行下载，并遵循该页面适用的说明和条款。

## 适用边界

DiffHallu 依赖同一模型的多次随机采样和生成概率信息。如果黑盒模型只能返回文本、不能提供等价的 teacher-forcing 评分，则不能完整复现该方法。当前默认句子切分方式和语义关系接口主要面向英文事实性内容。
