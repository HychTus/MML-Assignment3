"""
    Module contains final Model and all pieces of it.
"""
import os
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, GPT2LMHeadModel, GPT2Tokenizer


MODEL_PATH = "/data/chy/others/MML-Assignment3/models"

class ImageEncoder(nn.Module):
    """
    Encodes image and returns it's embedding.
    """

    def __init__(self, model, device="cpu"):
        super(ImageEncoder, self).__init__()

        self.device = device

        # from_pretrained 的 model 参数为 pretrained_model_name_or_path
        # from_pretrained 经过 @classmethod 修饰, 为类方法, 不是实例方法, 可以通过类名调用
        # from_pretrained 返回的是 CLIPProcessor 实例
        # Processor 对输入图像进行必要的预处理 (如调整大小、标准化)
        model_path = os.path.join(MODEL_PATH, model)
        self.preprocessor = CLIPProcessor.from_pretrained(model_path)
        
        # model 部分只使用 vison_model (类型为 nn.Module)
        self.model = CLIPModel.from_pretrained(model_path).vision_model.to(self.device)

    def forward(self, image):
        # only one image at a time
        #NOTE: 为什么每次只能处理一张图片? 这个效率有点低吧?
        image = self.preprocessor(images=image, return_tensors="pt").to(self.device)
        image_features = self.model(**image) # 将 dict 解包后作为关键字参数传入

        # 输出的 image_features 有多个 property (NOTE: 对应的类型是?)
        # pooler_output 通常是图像特征的高维表示, 是经过池化 (pooling) 的结果, 适合用于下游任务
        return image_features.pooler_output

        # CLIP vision model 对应 CLIPVisionTransformer


#TODO: 如果要调整成 Q-Former 应该如何处理?
class Mapping(nn.Module):
    """
    Maps image embedding to GPT-2 embedding.
    """

    def __init__(
        self,
        ep_len,
        num_layers,
        embed_size,
        output_embed_size,
        n_heads,
        forward_expansion,
        dropout,
        device="cpu",
    ):
        super(Mapping, self).__init__()

        self.ep_len = ep_len
        self.embed_size = embed_size
        self.output_embed_size = output_embed_size
        self.device = device
        
        # nn.TransformerEncoder 同样使用 nn.TransformerEncoderLayer 进行初始化
        # forward_expansion 表示 encoder block 的 MLP 需要进行几倍的扩展
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_size,
                nhead=n_heads,
                dim_feedforward=embed_size * forward_expansion,
                dropout=dropout,
                batch_first=True,
                device=device,
            ),
            num_layers=num_layers,
        ).to(self.device)

        # 从 embed_size 映射到 ep_len 个 token
        self.mapper = nn.Linear(embed_size, ep_len * output_embed_size).to(self.device)

        self.init_weights()

    def forward(self, img_embedded, train_mode=False):
        # 所以中间 attention 的作用是? 实际上只有一个 image token
        # 在这之前 img_embedded 不需要进行 view 吗?
        x = self.transformer_encoder(img_embedded)
        x = self.mapper(x)

        # * 有对于 [] 解包成位置参数的功能
        # train_mode 时对于批量进行操作, 否则只对于单个进行操作 (NOTE: 简化了?)
        x = x.view(
            *(
                [-1, self.ep_len, self.output_embed_size]
                if train_mode
                else [self.ep_len, self.output_embed_size]
            )
        )  # for batched input

        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

# 注意这里定义的是 text decoder
# image caption 中不存在 text encoder
class TextDecoder(nn.Module):
    """
    Processes embedding into caption.
    """

    def __init__(self, model, device="cpu"):
        super(TextDecoder, self).__init__()

        self.device = device

        # 加载 tokenizer, 并且将 tokenize 时用于填充长度的 pad_token 设置为 eos_token
        model_path = os.path.join(MODEL_PATH, model)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        #NOTE: GPT2LMHeadModel 是 GPT-2 模型的变种，带有语言模型头 (LM head)
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
        self.vocab_size = self.model.config.vocab_size

    def forward(self, embedding, attention_mask=None):
        # 根据 embedding 序列和 mask 来进行 transformer decode
        text_features = self.model(
            inputs_embeds=embedding, attention_mask=attention_mask
        )

        # logits 为对于每个 token 预测的分数, softmax 之后转转为概率分布
        return text_features.logits


# training.py 最终调用的是 Net class
class Net(nn.Module):
    """
    Final Model class. Puts all pieces together and generates caption based on image.
    """

    def __init__(
        self,
        clip_model,
        text_model,
        ep_len,
        num_layers,
        n_heads,
        forward_expansion,
        dropout,
        max_len,
        device="cpu",
    ):
        """
        Model constructor.
        Args:
            num_layers: number of layers in the TransformerEncoder
            n_heads: number of heads in the MultiHeadAttention
            forward_expansion: expansion factor for the feedforward layer
            dropout: dropout probability
            max_len: maximum length of the generated text
        """
        super(Net, self).__init__()

        self.device = device
        self.ep_len = ep_len

        # image encoder 根据 clip_model string 初始化, 需要指定 device
        self.ie = ImageEncoder(model=clip_model, device=device)

        # Mapping 过程中仍然维持了 image 的 hidden_size
        # 所以最终要保证输出的 emb_size 和 text_decoder 的 n_embd 一致
        self.mp = Mapping(
            ep_len=self.ep_len,
            num_layers=num_layers,
            embed_size=self.ie.model.config.hidden_size,
            n_heads=n_heads,
            forward_expansion=forward_expansion,
            dropout=dropout,
            device=device,
        )
        # text decoder 根据 text_model 初始化
        self.td = TextDecoder(model=text_model, device=device)

        # 不是很理解这部分为什么需要 matching
        # assert (
        #     self.ie.model.config.hidden_size == self.td.model.config.n_embd
        # ), "Embedding size of models mismatch"

        self.max_len = max_len

        # self.criterion = nn.CrossEntropyLoss(ignore_index=self.td.tokenizer.pad_token_id) # chanded on epoch 91
        self.criterion = nn.CrossEntropyLoss()

        self.freeze_layers()

    def freeze_layers(self):
        # 将 parameters 转化为 list, 使用 * 解包
        # 对于 Image Encoder 进行 Freezing
        # LLM Backbone 1st and last transformer layer 和 Projector 进行训练
        
        print(list(self.td.parameters()))

        for p in [
            *list(self.ie.parameters()),
            *list(self.td.parameters())[14:-14],
        ]:  # freeze everything, except 1st and last transformer layer in Decoder
            p.requires_grad = False
            # 不设置 requires_grad, optimizer 就不会更新

    # 分成 train_forward 和 forward 两个方法, 前者用于训练, 后者用于预测
    # 预测时只能使用一张图片 (为什么这样设置?)
    # 在 train_forward 时没有使用 self.ie, 在 forward 时使用了 self.ie


    def forward(self, img, temperature=1.0):
        """
        Caption generation for a single image.
        Args:
            img: image to generate caption for [PIL.Image]
        Returns:
            caption: generated caption [str]
            tokens: generated tokens [torch.Tensor]
        """

        if temperature <= 0.0:
            temperature = 1.0
            print("Temperature must be positive. Setting it to 1.0")

        with torch.no_grad():
            # 这里才使用了 image encoder
            img_embedded = self.ie(img)

            # (ep_len, embed_size)
            img_mapped = self.mp(img_embedded)

            sos_emb = self.td.model.transformer.wte(
                torch.tensor(self.td.tokenizer.bos_token_id).to(self.device)
            )

            # sos_emb shape embed_size -> (1, embed_size)
            sos_emb = sos_emb.unsqueeze(0)

            # (ep_len + 1, embed_size)
            start_emb = torch.cat([sos_emb, img_mapped], dim=0)

            tokens = []
            for _ in range(self.max_len):
                if len(tokens):
                    tok_emb = self.td.model.transformer.wte(
                        torch.tensor(tokens).to(self.device)
                    )

                    # 确实应该 visual token 在 text start token 后面
                    # 需要兼容 LLM backbone 的架构 (能不能放到前面去?)
                    # 在训练的时候是否添加了 start token? 没有添加吧?
                    # 由于是从后面开始生成的, 所以前面的影响不大?

                    # 训练时是 [visual token, text token] (text token 中包含了 <START>)
                    # 生成时是 [<START>, visual token, text token]
                    emb = torch.cat([start_emb, tok_emb], dim=0)
                else:
                    emb = start_emb

                # add positional enc
                pos_emb = self.td.model.transformer.wpe(
                    torch.arange(emb.shape[0]).to(self.device)
                )

                emb += pos_emb
                pred = self.td(emb)

                pred = torch.softmax(pred / temperature, dim=-1)

                _, pred = torch.max(pred, dim=1)

                last_token = pred[-1].item()

                tokens.append(last_token)

                if last_token == self.td.tokenizer.eos_token_id:
                    break

            decoded = self.td.tokenizer.decode(tokens[:-1])
            # 去除了最后的 end token, 并且只进行首字母大写
            decoded = decoded.strip()
            if len(decoded)>0:
                decoded = decoded[0].upper() + decoded[1:]

            return decoded, tokens

    def train_forward(self, img_emb, trg_cap, att_mask):
        # method should get embedded by CLIP images and trg_text without last token.
        # dataset should contain image, embedded image, text

        # 需要拆分 text 的输入和输出
        x, x_mask = trg_cap[:, :-1], att_mask[:, :-1]
        y = trg_cap[:, 1:]

        img_mapped = self.mp(img_emb, train_mode=True)

        # embed all texts and con cat with map sos
        # self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        # 输入的 x 是已经 tokenizer 的结果, 使用 text decoder 的 embedding layer
        text_emb = self.td.model.transformer.wte(x)

        # N, len, embed_size
        # print("shape::", img_mapped.shape, text_emb.shape)
        # torch.Size([4, 4, 768]) torch.Size([1, 39, 768]
        x = torch.concat([img_mapped, text_emb], dim=1)
        # 此处的 mask 表述的是有效 token 的范围
        # visual token 都是有效 token, 可以被 attention 使用
        #NOTE: 但是实际就是当成 attn_mask 使用的, 为什么呢?
        x_mask = torch.concat(
            [torch.ones(x_mask.shape[0], self.ep_len).to(self.device), x_mask], dim=1
        )

        # 使用 decoder 的相关结构进行 position embedding
        pos_emb = self.td.model.transformer.wpe(
            torch.arange(x.shape[1]).to(self.td.device)
        )
        pos_emb = pos_emb.expand_as(x)

        x += pos_emb

        res = self.td(x, attention_mask=x_mask)
        # res = torch.softmax(res, dim=2) # double softmax for ce_loss

        # criterion 中不需要忽视无效内容吗?
        loss = self.criterion(
            res[:, self.ep_len :, :].reshape(-1, res.shape[-1]), y.reshape(-1)
        )

        return loss

# 这代码写的是真狗屎, text 的相关处理实在是写得太烂了


if __name__ == "__main__":
    # 这部分应该是模型加载的测试代码, S 和 L 对应的组合
    for clip, text in [
        ["openai/clip-vit-base-patch32", "gpt2"],
        ["openai/clip-vit-large-patch14", "gpt2-medium"],
    ]:
        m = Net(
            clip_model=clip,
            text_model=text,
            ep_len=3,
            num_layers=6,
            n_heads=16,
            forward_expansion=4,
            dropout=0.1,
            max_len=20,
        )

        m.eval()
        r = m(torch.randn(3, 224, 224))
        print(r)

        m.train()
        N = 10
        emb = m.td.model.config.n_embd
        length = 20

        l = m.train_forward(
            torch.rand(N, emb),
            torch.randint(1, 50000, (N, length)),
            att_mask=torch.concat(
                [torch.ones(N, length - 3), torch.zeros(N, 3)], dim=1
            ),
        )
        print(l)

        # number of parameters
        print(f"Total number of parameters: {sum(p.numel() for p in m.parameters())}")
        print(
            f"Number of trainable parameters: {sum(p.numel() for p in m.parameters() if p.requires_grad)}"
        )
