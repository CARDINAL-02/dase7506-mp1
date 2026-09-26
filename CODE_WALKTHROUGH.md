# 代码逐步讲解：`rope.py` 与 `student.py`

> 这份文档是为了让你**完全拥有**自己的实现。所有形状和数值都是实际跑出来的，不是推的。
> 读完之后你应该能不看代码回答第 6 节的每一道自测题。
>
> **本文档里的每个数字都可以自己复核**：
>
> ```bash
> cd code && python ../trace_rope.py
> ```
>
> `trace_rope.py` 只读 `student.py` / `rope.py` 的类，不参与提交模型的推理路径。
> 如果哪个数字对不上，以脚本输出为准 —— 脚本是实测，文档是转述。

---

## 0. 阅读顺序

```
code/rope.py        28 行   ← 先读这个，机制本体，与模型无关
code/student.py     99 行   ← 再读这个，看机制插在哪里
code/model.py       68 行   ← 最后对照，确认 learned 档就是它
```

`student.py` 的 `GPT` 和 `Block` 是 `model.py` 的**改版**：算子序列基本照搬，只加了 `pos_mode` 分支和 RoPE 调用。所以"哪里不一样"比"每一行是什么"更重要。

---

## 1. `rope.py` — 机制本体

### 1.1 先理解目标

注意力分数是 $q\cdot k$。我们希望这个分数**只取决于两个 token 相隔多远**，而不是它们各自在第几号位置。

RoPE 的做法：把 q 和 k 各自按**自己的绝对位置**旋转一个角度。因为旋转可以复合、且旋转的逆就是反向旋转：

$$(R(m)q)\cdot(R(n)k)=q^{\top}\underbrace{R(m)^{\top}R(n)}_{=\,R(n-m)}k=q^{\top}R(n-m)k$$

绝对位置 $m,n$ 抵消，**只剩差 $n-m$**。这就是全部原理。

实测验证（`head_dim=8`，固定 q、k，只挪位置）：

| (m, n) | gap | 分数 |
|---|---:|---:|
| (0,0) (1,1) (5,5) | 0 | −0.526053（三个完全相同） |
| (0,1) (4,5) (10,11) | 1 | 0.651395（三个完全相同） |
| (0,7) (20,27) | 7 | 0.133757（两个完全相同） |

而且旋转是正交变换，**保持向量长度**：`||q|| = 3.395326`，旋转后仍是 `3.395326 / 3.395327 / 3.395327`。这很重要 —— 它意味着 RoPE 不会放大注意力 logits，训练稳定性不受影响。

### 1.2 `build_rope` 逐行

```python
def build_rope(head_dim, context, base=10000.0):
    if head_dim % 2:
        raise ValueError(f'RoPE needs an even head_dim, got {head_dim}.')
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(context).float(), inv_freq)
    return freqs.cos(), freqs.sin()
```

**第 1 行 `if head_dim % 2: raise`**
RoPE 要成对旋转，奇数维度无法两两配对。属于防御性检查 —— 我们的 `width=128, heads=4` 给出 `head_dim=32`，永远触发不到，但万一有人改了 config 会立刻报错而不是静默出错。

**第 2 行，拆成四步看**（`head_dim=32` 实测值）：

| 子表达式 | 结果 |
|---|---|
| `torch.arange(0, 32, 2)` | shape `(16,)`，值 `[0, 2, 4, ..., 30]` |
| `... / head_dim` | `[0.0, 0.0625, 0.125, 0.1875, ..., 0.9375]` |
| `10000 ** 那个` | `[1, 1.778, 3.162, 5.623, ..., 5623.4]` |
| `1.0 / 那个` | **`inv_freq[0] = 1.000000`**，**`inv_freq[-1] = 0.00017783`** |

`inv_freq[j]` 是"第 j 组 pair 每前进一个 token 转多少弧度"。

- `inv_freq[0] = 1` → 每个 token 转 1 弧度。两个相邻 token 之间就转了 57°，**高频**，能分辨邻近位置。
- `inv_freq[15] = 1.78e-4` → 1000 个 token 才转 0.18 弧度。**低频**，编码长程位置。

这是刻意的设计：**16 组 pair 覆盖 16 个不同的距离尺度**，类似傅里叶基。任何单一频率都无法同时分辨"相隔 1"和"相隔 200"。

**第 3 行 `torch.outer(...)`**
`arange(context)` shape `(256,)`，`inv_freq` shape `(16,)` → `freqs` shape **`(256, 16)`**，语义是 `freqs[位置][pair组] = 该位置该转的角度`。

实测前几行：
```
freqs[0]   = [0.0, 0.0, 0.0, 0.0, ...]      ← 位置 0 角度全为 0
freqs[1]   = [1.0, 0.5623, 0.3162, 0.1778, ...]
freqs[100] = [100.0, 56.2341, 31.6228, 17.7828, ...]
```

> **注意 `freqs[0]` 全为 0** —— 位置 0 的角度是 0，所以 `cos=1, sin=0`，旋转是恒等变换。这意味着**序列第一个 token 的 q、k 完全不被改动**。这是正确行为（它没有"前面"可参照），也解释了下面 1.3 的实测现象。

**第 4 行** 返回 `cos, sin`，各 shape `(256, 16)`。

> **为什么 `head_dim` 必须作为参数传入、不能写死 32？**
> 契约测试用 `dict(vocab=2048, width=32, heads=4, depth=2, context=256)` → `head_dim = 32//4 = 8`。写死 32 的话 `inv_freq` 有 16 项，而测试里 q 的末维只有 8，`apply_rope` 里 `x[..., 0::2]` 得到 4 列 —— 广播时形状冲突，直接崩。**这是整个实现里最真实的契约陷阱。**

### 1.3 `apply_rope` 逐行

```python
def apply_rope(x, cos, sin):
    cos, sin = cos[:x.shape[-2]].to(x.dtype), sin[:x.shape[-2]].to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
```

**第 1 行做两件事：**

1. **`[:x.shape[-2]]` 截到实际序列长度。** 表是按 `context=256` 预建的，但：
   - 契约测试序列长度只有 **12**
   - 评测的最后一个窗口是**短窗口**（`common.windows()` 会补零）
   - 训练时是满 256
   截断让同一个表服务所有长度。

2. **`.to(x.dtype)`。** 训练走 BF16（`--precision auto` 在 CUDA 上选 bf16），表是 fp32。不转会类型不匹配。

**第 2 行** 把末维 32 切成两个 16 列：
```
x[..., 0::2]  → 取第 0,2,4,...,30 列    shape (1,1,6,16)
x[..., 1::2]  → 取第 1,3,5,...,31 列    shape (1,1,6,16)
```
也就是把 32 维拆成 16 组**相邻** pair：`(0,1), (2,3), ..., (30,31)`。

**第 3 行** 就是二维旋转公式：

$$\begin{bmatrix}e'\\o'\end{bmatrix}=\begin{bmatrix}e\cos\theta-o\sin\theta\\ e\sin\theta+o\cos\theta\end{bmatrix}$$

- `torch.stack((..., ...), dim=-1)` → shape `(1,1,6,16,2)`，最后一维是 `(新偶位, 新奇位)`
- `.flatten(-2)` → `(1,1,6,32)`，把 16 组 pair 重新压成交错排布

实测（`[1,1,6,32]` 输入）：
```
x[0,0,0,:4]   = [-1.12584, -1.15236, -0.250579, -0.433879]
out[0,0,0,:4] = [-1.12584, -1.15236, -0.250579, -0.433879]   ← 位置 0，原样不变
x[0,0,3,:4]   = [ 1.645867, -1.360169, 0.344565, 0.519868]
out[0,0,3,:4] = [-1.437449,  1.578822, -0.556318, 0.281954]  ← 位置 3，被旋转了
```

长度保持（每个位置逐 token 验证）：
```
in : 5.691975  6.137256  5.860567  5.838870  4.173726  4.791229
out: 5.691975  6.137256  5.860567  5.838871  4.173726  4.791229
```
第 4 个位置的差异是 `1e-6` 量级的浮点误差，正常。

> **配对约定可以自由选，但 q 和 k 必须用同一种。**
>
> 文献里两种排布都有人用：本文的**相邻配对** `(0,1),(2,3)…`，和原始 RoFormer 的**前后对半** `(0,16),(1,17)…`（后者需要把 cos/sin 复制成 d 维）。实测两者都给出"只依赖 gap"的性质 —— 所以**配对方式本身不是陷阱**，它只是对 head 维度做了一个固定的置换。
>
> **真正会静默失效的是只转 q、不转 k。** 此时分数变成 $q^\top R(m)^\top k$，仍然含绝对位置 $m$。实测同 gap 不同绝对位置的分数离散度：
>
> | 做法 | 离散度 | |
> |---|---:|---|
> | 相邻配对，q 和 k 都转（本文实现） | 1e-6 | 相对位置成立 |
> | 前后对半配对，q 和 k 都转 | 1e-6 | 相对位置成立 |
> | **只转 q，k 原样** | **3.85** | **静默失效** |
>
> 第三种**不报错、不崩溃**，只是模型悄悄学不到相对位置。这是最危险的一类 bug。
>
> 排查方法：固定 q、k，检验同 gap 不同绝对位置的分数是否相同（就是 1.1 那张表）。

### 1.4 为什么不会泄漏未来信息

旋转是**逐位置独立**的：位置 t 的 q 只用到了位置 t 自己的索引，不与其他任何位置交互。所以 RoPE 本身不产生跨位置混合，**因果性完全由后面的 `is_causal=True` 保证**。

契约测试 `test_future_inputs_cannot_change_earlier_predictions` 直接验证：改动 `x[:, 7:]` 后，前 7 个位置的输出必须逐位不变。

---

## 2. `student.py` — 机制插在哪里

### 2.1 `Block.__init__`

```python
def __init__(self, width, heads, pos_mode):
    self.heads = heads
    self.head_dim = width // heads
    self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
    self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
    self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
```

和 `model.py` 的 `Block` **逐行相同**，唯一区别是 `__init__` 多收一个 `pos_mode` 参数（实际没用到，因为 cos/sin 是从外面传进来的）。

三处合并进一个 `qkv` 线性层是原版的设计：`Linear(128, 384)` 一次算出 q、k、v 三个 128 维向量。

### 2.2 `Block.forward` 逐步（实测 `B=2, T=5`）

```python
batch, length, width = x.shape
q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
```

| 表达式 | 形状 |
|---|---|
| `x` | `(2, 5, 128)` |
| `norm1(x)` | `(2, 5, 128)` |
| `qkv(...)` | `(2, 5, 384)` |
| `.view(B, T, 3, heads, head_dim)` | `(2, 5, 3, 4, 32)` |
| `.permute(2, 0, 3, 1, 4)` | **`(3, 2, 4, 5, 32)`** = `[3, B, H, T, D]` |
| 解包 `q, k, v` | 各 `(2, 4, 5, 32)` = `[B, H, T, D]` |

`permute` 把"3"（qkv）和"heads"（4）提到前面，是为了让 `scaled_dot_product_attention` 拿到它要的 `[B, H, T, D]` 布局。

```python
if cos is not None:
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
```

**三个设计决定：**

**① 为什么在 `Block` 里加，不在 `features()` 里？**
q 和 k 只在 block 内部存在（`features` 里只有 token 嵌入，还没投影）。放在这里的好处是 `forward()`（训练）和 `predict_log_probs()`（评测）**两条路径自动都走到**，因为二者都经过 `features → block`。不用维护两份代码、不会漏。

**② 为什么只转 q 和 k，不转 v？**
注意力分数是 $q^\top k$ —— 位置决定的是"**该看哪里**"（query 与 key 的匹配）。v 是"看到之后**搬运什么内容**"。给 v 加一个位置相关的旋转，只会让输出多一个旋转，而紧接着的 `proj` 线性层本来就能吸收掉。所以转 v 无收益，标准做法也不转。

**③ 为什么在 SDPA 之前？**
RoPE 逐位置操作、不改变序列结构，所以放在因果掩码之前还是之后**对正确性没有影响**。放前面是因为 PyTorch 的 fused attention 把旋转和掩码揉在一起、没法插进去。

```python
attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))
return x + self.mlp(self.norm2(x))
```

| 表达式 | 形状 |
|---|---|
| `attended` | `(2, 4, 5, 32)` |
| `.transpose(1, 2)` | `(2, 5, 4, 32)` = `[B, T, H, D]` |
| `.reshape(B, T, width)` | `(2, 5, 128)` —— 把 4 个 head 拼回去 |
| `proj(...)` | `(2, 5, 128)` |
| `x + ...` | `(2, 5, 128)` ← 残差连接 1 |
| `x + mlp(norm2(x))` | `(2, 5, 128)` ← 残差连接 2，block 输出 |

`is_causal=True` 的实测效果（`B=0, H=0` 的原始 $q k^\top$ 矩阵，行=query 位置，列=key 位置）：

```
                    key 位置
              0       1       2       3       4
query  0    0.146    MASK    MASK    MASK    MASK
位置   1    0.159    0.217   MASK    MASK    MASK
       2   -0.500   -0.209    0.026   MASK    MASK
       3   -0.046    0.070   -0.295    0.060   MASK
       4   -0.109    0.032   -0.001   -0.047  -0.167
```

上三角全被屏蔽，**位置 t 只能看到 ≤ t**。这就是因果性的实现处。

### 2.3 `GPT.__init__` 的三种分支

```python
self.pos_mode = config.get('pos_mode', 'rope')
if self.pos_mode not in POS_MODES:
    raise ValueError(...)
self.token = nn.Embedding(config['vocab'], width)
if self.pos_mode == 'learned':
    self.pos = nn.Embedding(self.context, width)
self.blocks = nn.ModuleList([Block(width, config['heads'], self.pos_mode) for _ in range(config['depth'])])
self.norm = nn.LayerNorm(width)
self.head = nn.Linear(width, config['vocab'], bias=False)    # [B,T,128] -> [B,T,2048]
self.apply(self.initialize)
self.head.weight = self.token.weight                         # ← 权重绑定
```

三个模式实测对照：

| 档 | 参数量 | 有 `pos` 嵌入 | 有 rope buffer | `state_dict` 张量数 |
|---|---:|---|---|---:|
| `learned` | **1,088,256** | ✅ | ❌ | **53** |
| `none` | **1,055,488** | ❌ | ❌ | **52** |
| `rope` | **1,055,488** | ❌ | ✅ | **52** |

- 差值 `32,768 = 256 × 128`，正好是那张位置表。
- `none` 和 `rope` **参数量完全相同** —— 这是报告 §1.3 里那个干净对照的根据。
- `rope` 的 buffer 不计入 `state_dict`（见 2.5），所以是 52 不是 54。

`self.head.weight = self.token.weight` 是**权重绑定**（weight tying）：输入嵌入矩阵和输出投影矩阵共享同一份参数。这是原版就有的设计，也是 `learned` 档比 `rope` 多的那 32,768 参数之外、两档共有的结构。

### 2.4 `features` / `forward` / `predict_log_probs`

```python
def features(self, ids):
    x = self.token(ids)
    if self.pos_mode == 'learned':
        x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
    cos = self.rope_cos if self.pos_mode == 'rope' else None
    sin = self.rope_sin if self.pos_mode == 'rope' else None
    for block in self.blocks:
        x = block(x, cos, sin)
    return self.norm(x)
```

三档在这里分岔：

- **`learned`**：`x = token(ids) + pos(arange(T))` —— 位置是**加**上去的
- **`none`**：什么都不加，cos/sin 传 `None` → block 跳过旋转
- **`rope`**：不加任何东西，但把 cos/sin 传下去 → 每个 block 里**旋转** q、k

注意 `learned` 档的这行与 `model.py` 的
```python
x = self.token(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
```
是**完全相同的算子序列**（只是拆成了两行）。

> **这是整个消融成立的前提。** 我们验证过三层：
> 1. `state_dict` 的键集合完全相同
> 2. 加载 `model.py` 的权重后，输出 `torch.equal` 为真（**逐位相同**）
> 3. 训练后 validation BPB 在两个训练长度上都精确相同：**2.0729**（1200 步）和 **1.7532**（4800 步）
>
> 所以三档之间**只差位置信息的传递方式**，没有别的混杂因素。

```python
def forward(self, ids):
    """训练接口：未归一化的 logits"""
    return self.head(self.features(ids))

def predict_log_probs(self, ids):
    """评测接口：归一化的自然对数概率"""
    return F.log_softmax(self(ids).float(), dim=-1)
```

`.float()` 是**必须的**：`features` 的输出可能是 BF16，而评分器要求 log-prob 满足 `|logsumexp| < 1e-3` 的归一化检查 —— BF16 精度不够。先升到 FP32 再做 softmax。

`predict_log_probs` 是**无状态**的：每次调用都从 `ids` 重新算一遍，不缓存任何东西。所以每个评测窗口天然是全新的，满足契约的 `test_state_resets_between_windows`。

### 2.5 那个 `persistent=False`

```python
if self.pos_mode == 'rope':
    cos, sin = build_rope(width // config['heads'], self.context)
    self.register_buffer('rope_cos', cos, persistent=False)
    self.register_buffer('rope_sin', sin, persistent=False)
```

- **用 buffer 而不是普通属性**：`.to(device)` / `.cpu()` 会自动搬运。训练上 GPU、存档时 `model.cpu()`，都不用管。
- **`persistent=False`**：不进 `state_dict()`。**它们是从 config 推导出来的常量**，没必要存 —— 加载时 `build_rope` 会重新算出完全一样的表。
- **它们是常量、不随调用变化** → 不构成跨窗口状态 → `test_state_resets_between_windows` 通过。

`width // config['heads']` 就是 `head_dim`（128 // 4 = 32），**由 config 推出而非写死** —— 同 1.2 节的契约陷阱。

---

## 3. 完整数据流一页图

```
ids  [B, T]
 │
 ├─ token(ids)                                  [B, T, 128]
 │
 ├─ [learned] + pos(arange(T))                  [B, T, 128]   ← 只有 learned 档
 │
 └─ for each of 4 blocks:
      x ──► norm1 ──► qkv ──► view/permute ──► q,k,v  [B, 4, T, 32]
                                                │
                            [rope] apply_rope(q), apply_rope(k)  ← 只有 rope 档
                                                │
                          scaled_dot_product_attention(is_causal=True)
                                                │
                              transpose/reshape ──► proj
                                                │
      x = x + proj(...)                         │  ← 残差 1
      x = x + mlp(norm2(x))                     │  ← 残差 2
 │
 ├─ norm                                        [B, T, 128]
 └─ head (绑定 token 权重) ──► logits            [B, T, 2048]

评测时： F.log_softmax(logits.float(), -1) ──► [B, T, 2048] 归一化对数概率
```

---

## 4. 你被追问时的一句话版本

| 问题 | 一句话回答 |
|---|---|
| RoPE 干什么？ | 把 q、k 按各自位置旋转，使 $q\cdot k$ 只依赖位置差 $n-m$，因为 $R(m)^\top R(n)=R(n-m)$ |
| 为什么因果安全？ | 旋转逐位置独立、不跨位置混合；因果性由 `is_causal=True` 保证 |
| 为什么只转 q、k？ | 位置决定"看哪里"（q·k 匹配），不决定"搬什么"（v）；转 v 会被 `proj` 吸收 |
| 收益是省了 32k 参数吗？ | 不是。`none` 与 `rope` 参数量完全相同，差 0.1313 BPB |
| `learned` 档真等于基线吗？ | 是。键集合相同、加载同权重后输出逐位相同、两个长度的 BPB 都精确相同 |
| 表为什么不存进 checkpoint？ | 由 config 推出的常量，加载时重算，`persistent=False` |
| 为什么 `head_dim` 从参数推？ | 契约测试用 `head_dim=8`，写死 32 会形状冲突 |

---

## 5. 自己动手验证

```bash
cd MP1_student_starter/code

# 重新生成本文档引用的全部数字（形状、频率、分数、掩码、参数量）
python ../trace_rope.py

# 契约测试（会走 rope 路径，因为默认 pos_mode='rope'）
python -m unittest discover -s tests -v

# 三档参数量 + learned 与基线的逐位一致性
python -c "
import json, torch, model, student
cfg = json.load(open('configs/baseline.json'))
for m_ in student.POS_MODES:
    print(m_, sum(p.numel() for p in student.build_model(dict(cfg,pos_mode=m_)).parameters()))
torch.manual_seed(0); a = model.build_model(cfg).eval()
torch.manual_seed(0); b = student.build_model(dict(cfg,pos_mode='learned')).eval()
b.load_state_dict(a.state_dict())
x = torch.randint(0,2048,(2,64))
with torch.no_grad(): print('learned == model.py :', torch.equal(a(x), b(x)))
"
```

---

## 6. 自测题（能全部答上来就算掌握了）

1. `build_rope(8, 16)` 返回的两个张量各是什么形状？为什么是 16 而不是 8？
2. `inv_freq[0] = 1.0` 意味着位置 1 和位置 2 之间的旋转角是多少度？
3. 为什么 `apply_rope` 里要写 `cos[:x.shape[-2]]` 而不是直接用 `cos`？
4. 如果只对 q 调用 `apply_rope`、不对 k 调用，会怎样？代码会报错吗？
5. `none` 档和 `rope` 档的 `state_dict` 键集合相同吗？为什么？
6. 位置 0 的 q、k 经过 `apply_rope` 后会不会改变？为什么？
7. `predict_log_probs` 里为什么要 `.float()`？
8. 为什么 RoPE 加在 `Block` 里而不是 `features()` 里？
9. `rope_cos` 用 `persistent=False` 注册，对 `evaluate.py` 加载 checkpoint 有影响吗？
10. 报告里"`rope` vs `none` 是唯一干净的对照"这句话，代码上的根据是哪一行？

<details>
<summary>参考答案要点</summary>

1. `(16, 4)` 各 —— 因为 `head_dim // 2 = 4`，`arange(0,8,2)` 只有 4 项。
2. 1 弧度 ≈ 57.3°。
3. 表按 `context=256` 预建，但契约测试用 T=12、评测有短窗口，必须截到实际长度。
4. 不会报错，但**相对位置性质被破坏** —— 分数变成 $q^\top R(m)^\top k$，仍依赖绝对位置 $m$。实测同 gap 不同绝对位置的分数离散度从 1e-6 跳到 **3.85**。代码静默失效、不崩溃，是最难查的一类 bug。
5. 相同（都是 52 个）—— 因为 `rope` 的表是 `persistent=False`，不进 `state_dict`；两档的结构差异只在 buffer 而不在参数。
6. 不会。`freqs[0]` 全为 0 → `cos=1, sin=0` → 旋转是恒等变换。
7. `features` 的输出可能是 BF16，评分器要求 `|logsumexp| < 1e-3`，BF16 精度不够。
8. q、k 只在 block 内部存在；放在这里，`forward` 和 `predict_log_probs` 两条路径自动都覆盖到。
9. 没有影响。加载时 `__init__` 会依据 checkpoint 里的 config 重新 `build_rope` 算出同样的表。
10. `if self.pos_mode == 'learned': self.pos = nn.Embedding(...)` 这一行 —— 因为只有 `learned` 档创建位置表，`none` 和 `rope` 的参数集完全相同。

</details>
