"""
    Module contains final Model and all pieces of it.
"""
import os
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, Qwen2Tokenizer, Qwen2ForCausalLM, Blip2QFormerAttention



MODEL_PATH = "/data/chy/others/MML-Assignment3/models"

class ImageEncoder(nn.Module):
    """
    Encodes image and returns it's embedding.
    """

    def __init__(self, model, device="cpu"):
        super(ImageEncoder, self).__init__()

        self.device = device
        model_path = os.path.join(MODEL_PATH, model)
        self.preprocessor = CLIPProcessor.from_pretrained(model_path)
        self.model = CLIPModel.from_pretrained(model_path).vision_model.to(self.device)

    def forward(self, image):
        image = self.preprocessor(images=image, return_tensors="pt").to(self.device)
        image_features = self.model(**image)
        return image_features.pooler_output


class QFormerDecoderLayer(nn.Module):
    def __init__(
            self, 
            embed_size, 
            num_heads, 
            dropout, 
            device
        ):
        super(QFormerDecoderLayer, self).__init__()
        self.device = device

        # 自注意力
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_size, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_size)
        self.dropout1 = nn.Dropout(dropout)

        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_size, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout2 = nn.Dropout(dropout)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.GELU(),
            nn.Linear(embed_size * 4, embed_size),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, queries, img_embeddings, self_attn_mask=None):
        # 自注意力
        attn_output, _ = self.self_attn(
            query=queries, 
            key=queries, 
            value=queries, 
            attn_mask=self_attn_mask
        )
        queries = queries + self.dropout1(attn_output)
        queries = self.norm1(queries)

        # 交叉注意力
        attn_output, _ = self.cross_attn(
            query=queries, 
            key=img_embeddings, 
            value=img_embeddings
        )
        queries = queries + self.dropout2(attn_output)
        queries = self.norm2(queries)

        # 前馈网络
        ffn_output = self.ffn(queries)
        queries = queries + self.dropout3(ffn_output)
        queries = self.norm3(queries)

        return queries


class QFormerMapping(nn.Module):
    def __init__(
            self, 
            embed_size,
            output_embed_size, 
            num_heads, 
            dropout, 
            num_queries, 
            num_layers,
            device
        ):
        super(QFormerMapping, self).__init__()
        self.device = device
        self.num_queries = num_queries
        self.query_embeddings = nn.Parameter(torch.randn(1, num_queries, embed_size).to(device))
        
        # 定义多层解码器层
        self.layers = nn.ModuleList([
            QFormerDecoderLayer(
                embed_size=embed_size, 
                num_heads=num_heads, 
                dropout=dropout, 
                device=device
            )
            for _ in range(num_layers)
        ])
        
        # 最终的映射层
        self.mapper = nn.Linear(embed_size, output_embed_size)
        self.init_weights()


    def forward(self, img_embeddings, self_attn_mask=None):
        # self_attn_mask 在 QFormer 中理论上应该不用使用
        # img_embeddings: [batch_size, seq_len, embed_size]
        batch_size = img_embeddings.size(0)
        queries = self.query_embeddings.expand(batch_size, self.num_queries, -1)
        queries = queries.to(self.device)
        img_embeddings = img_embeddings.to(self.device)
        
        for layer in self.layers:
            queries = layer(queries, img_embeddings, self_attn_mask)
        
        mapped_embeddings = self.mapper(queries)
        return mapped_embeddings


    def init_weights(self):
        # 这部分代码不需要修改, 已经能够处理 transformer 的初始化情况
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class TextDecoder(nn.Module):
    """
    Processes embedding into caption.
    """

    def __init__(self, model, device="cpu"):
        super(TextDecoder, self).__init__()

        self.device = device
        model_path = os.path.join(MODEL_PATH, model)
        self.tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = Qwen2ForCausalLM.from_pretrained(model_path).to(self.device)
        self.vocab_size = self.model.config.vocab_size

        self.word_embeddings = self.model.get_input_embeddings()
    

    def forward(self, embedding, attention_mask=None):
        model_output = self.model(
            inputs_embeds=embedding, attention_mask=attention_mask
        )
        return model_output.logits


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
        num_heads,
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

        self.ie = ImageEncoder(model=clip_model, device=device)
        self.td = TextDecoder(model=text_model, device=device)

        self.mp = QFormerMapping(
            embed_size=self.ie.model.config.hidden_size,
            output_embed_size=self.td.model.config.hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            num_queries=ep_len, # query 数量就对应最终的 token 个数
            num_layers=num_layers,
            device=device,
        )

        self.max_len = max_len
        self.criterion = nn.CrossEntropyLoss()
        self.freeze_layers()

    def freeze_layers(self):
        #TODO: 这里针对 Qwen 需要进行修改
        # for name, param in self.td.named_parameters():
        #     print(name)

        for p in [
            *list(self.ie.parameters()),
            *list(self.td.parameters())[13:-13],
            list(self.td.parameters())[0],  # model.model.embed_tokens.weight
            list(self.td.parameters())[-1], # model.model.norm.weight
        ]:  # freeze everything, except 1st and last transformer layer in Decoder
            p.requires_grad = False


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
            img_embedded = self.ie(img)
            img_mapped = self.mp(img_embedded)
            start_emb = img_mapped

            tokens = []
            for _ in range(self.max_len):
                if len(tokens):
                    tok_emb = self.td.word_embeddings(torch.tensor(tokens).to(self.device))
                    emb = torch.cat([start_emb, tok_emb], dim=0)
                else:
                    emb = start_emb

                # print("size::", emb.size()) # torch.Size([4, 896])
                pred = self.td(emb.unsqueeze(0))
                pred = pred.squeeze(0)

                pred = torch.softmax(pred / temperature, dim=-1)

                _, pred = torch.max(pred, dim=1)

                last_token = pred[-1].item()

                tokens.append(last_token)

                if last_token == self.td.tokenizer.eos_token_id:
                    break

            decoded = self.td.tokenizer.decode(tokens[:-1])
            decoded = decoded.strip()
            if len(decoded)>0:
                decoded = decoded[0].upper() + decoded[1:]

            return decoded, tokens

    def train_forward(self, img_emb, trg_cap, att_mask):
        # method should get embedded by CLIP images and trg_text without last token.
        # dataset should contain image, embedded image, text

        x, x_mask = trg_cap[:, :-1], att_mask[:, :-1]
        y = trg_cap[:, 1:]

        img_mapped = self.mp(img_emb, train_mode=True)
        text_emb = self.td.word_embeddings(x)
        x = torch.concat([img_mapped, text_emb], dim=1)
        x_mask = torch.concat(
            [torch.ones(x_mask.shape[0], self.ep_len).to(self.device), x_mask], dim=1
        )

        res = self.td(x, attention_mask=x_mask)
        loss = self.criterion(
            res[:, self.ep_len :, :].reshape(-1, res.shape[-1]), y.reshape(-1)
        )

        return loss


if __name__ == "__main__":
    for clip, text in [
        # ["openai/clip-vit-base-patch32", "gpt2"],
        # ["openai/clip-vit-large-patch14", "gpt2-medium"],
        ["clip-vit-base-patch32", "Qwen2.5-0.5B"],
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


"""
Qwen2Model(
  (embed_tokens): Embedding(151936, 896)
  (layers): ModuleList(
    (0-23): 24 x Qwen2DecoderLayer(
      (self_attn): Qwen2SdpaAttention(
        (q_proj): Linear(in_features=896, out_features=896, bias=True)
        (k_proj): Linear(in_features=896, out_features=128, bias=True)
        (v_proj): Linear(in_features=896, out_features=128, bias=True)
        (o_proj): Linear(in_features=896, out_features=896, bias=False)
        (rotary_emb): Qwen2RotaryEmbedding()
      )
      (mlp): Qwen2MLP(
        (gate_proj): Linear(in_features=896, out_features=4864, bias=False)
        (up_proj): Linear(in_features=896, out_features=4864, bias=False)
        (down_proj): Linear(in_features=4864, out_features=896, bias=False)
        (act_fn): SiLU()
      )
      (input_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
      (post_attention_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
    )
  )
  (norm): Qwen2RMSNorm((896,), eps=1e-06)
  (rotary_emb): Qwen2RotaryEmbedding()
)

GPT2Model(
  (wte): Embedding(50257, 768)
  (wpe): Embedding(1024, 768)
  (drop): Dropout(p=0.1, inplace=False)
  (h): ModuleList(
    (0-11): 12 x GPT2Block(
      (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (attn): GPT2SdpaAttention(
        (c_attn): Conv1D(nf=2304, nx=768)
        (c_proj): Conv1D(nf=768, nx=768)
        (attn_dropout): Dropout(p=0.1, inplace=False)
        (resid_dropout): Dropout(p=0.1, inplace=False)
      )
      (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (mlp): GPT2MLP(
        (c_fc): Conv1D(nf=3072, nx=768)
        (c_proj): Conv1D(nf=768, nx=3072)
        (act): NewGELUActivation()
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
)
"""