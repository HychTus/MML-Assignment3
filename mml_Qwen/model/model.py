"""
    Module contains final Model and all pieces of it.
"""
import os
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, Qwen2Tokenizer, Qwen2ForCausalLM

# 调整成使用 Qwen2Tokenizer 和 Qwen2ForCausalLM
# Qwen2Model 计算出来的是 hidden state, Qwen2ForCausalLM 才能得到 logits


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

        # 从 embed_size 映射到 ep_len 个 output_embed_size token
        self.mapper = nn.Linear(embed_size, ep_len * output_embed_size).to(self.device)
        self.init_weights()

    def forward(self, img_embedded, train_mode=False):
        x = self.transformer_encoder(img_embedded)
        x = self.mapper(x)

        #BUG: 这里需要修改为 output_embed_size
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


class TextDecoder(nn.Module):
    """
    Processes embedding into caption.
    """

    def __init__(self, model, device="cpu"):
        super(TextDecoder, self).__init__()

        # 调整为使用 AutoTokenize, pad_token 和 eos_token 的设置仍然不变
        self.device = device
        model_path = os.path.join(MODEL_PATH, model)
        self.tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 从 GPT2LMHeadModel 调整为使用 AutoModel
        self.model = Qwen2ForCausalLM.from_pretrained(model_path).to(self.device)
        self.vocab_size = self.model.config.vocab_size

        self.word_embeddings = self.model.get_input_embeddings()
        # for param in self.word_embeddings.parameters():
        #     param.requires_grad = False
        # 因为是共享的, 所以不用单独去冻结?

    def forward(self, embedding, attention_mask=None):
        # 根据 embedding 序列和 mask 来进行 transformer decode
        # 这里强制使用的是 inputs_embeds, 可以选择使用 input_ids
        model_output = self.model(
            inputs_embeds=embedding, attention_mask=attention_mask
        )
        # BUG: AttributeError: 'BaseModelOutputWithPast' object has no attribute 'logits'
        # 原本使用的是 text_features.logits, 对应预测的分数

        # print("type::", type(model_output)) 
        # <class 'transformers.modeling_outputs.BaseModelOutputWithPast'>
        
        # print("keys::", model_output.keys())
        # keys:: odict_keys(['last_hidden_state', 'past_key_values'])
        
        # print("size::", model_output['last_hidden_state'].size())
        # torch.Size([64, 43, 896])

        # last_hidden_state 为 decode 之后整个序列最后的隐藏层
        # 这里输出的不是最终 vcab_size 的 logits, 所以出现了问题!
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
        # print("cuda_device", os.environ["CUDA_VISIBLE_DEVICES"])

        self.device = device
        self.ep_len = ep_len

        self.ie = ImageEncoder(model=clip_model, device=device)
        self.td = TextDecoder(model=text_model, device=device)

        self.mp = Mapping(
            ep_len=self.ep_len,
            num_layers=num_layers,
            embed_size=self.ie.model.config.hidden_size,
            output_embed_size=self.td.model.config.hidden_size,
            n_heads=n_heads,
            forward_expansion=forward_expansion,
            dropout=dropout,
            device=device,
        )
        
        self.max_len = max_len
        self.criterion = nn.CrossEntropyLoss()
        self.freeze_layers()

    def freeze_layers(self):
        #TODO: 这里针对 Qwen 需要进行修改
        for name, param in self.td.named_parameters():
            print(name)

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

            # (ep_len, embed_size)
            img_mapped = self.mp(img_embedded)

            # sos_emb = self.td.word_embeddings(torch.tensor(self.td.tokenizer.bos_token_id).to(self.device))

            # sos_emb shape embed_size -> (1, embed_size)
            # sos_emb = sos_emb.unsqueeze(0)

            # (ep_len + 1, embed_size)
            # start_emb = torch.cat([sos_emb, img_mapped], dim=0)
            start_emb = img_mapped
            # BUG: Qwen2Tokenizer 没有 bos token

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
        # print("img_emb::", img_mapped.size()) torch.Size([64, 4, 896])

        # word embedding
        # word_embeddings = self.td.model.embed_tokens
        # x = word_embeddings(x.long())
        # 直接取出对应的层不行, 还是应该通过方法来获取
        # 之前出现类型错误, 要求输出 long 但是输出 

        # word embedding
        # print("grad::", list(self.td.word_embeddings.parameters())[0].requires_grad) False
        text_emb = self.td.word_embeddings(x)
        # print("text_emb::", text_emb.size()) # torch.Size([64, 39, 896])
        x = torch.concat([img_mapped, text_emb], dim=1)
        # print("x::", x.size()) torch.Size([64, 43, 896])

        x_mask = torch.concat(
            [torch.ones(x_mask.shape[0], self.ep_len).to(self.device), x_mask], dim=1
        )
        # print("x_mask::", x_mask.size()) torch.Size([64, 43])
        # 这里使用的 mask 是正常的, 表示哪些是有效的 token

        res = self.td(x, attention_mask=x_mask)
        # print("res::", res.size()) torch.Size([64, 43, 896])
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