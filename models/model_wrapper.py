"""
Enhanced Model Wrapper V2 with AirLLM Support

Unified interface for different LLM APIs and local models
- Full attention/layer access for HuggingFace models
- AirLLM support for very large models (70B+) on limited VRAM
- API models (OpenAI, Anthropic, Google, xAI)

AirLLM enables running models like:
- DeepSeek-R1 (671B) on 4GB VRAM
- Llama-3.1-405B on 8GB VRAM
- Any model that doesn't fit in memory
"""

import os
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
from abc import ABC, abstractmethod
import warnings

# API imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai package not installed")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic package not installed")

try:
    from google import genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print("Warning: google-genai package not installed")

MISTRAL3_PATTERNS = ("ministral-3", "mistral-3", "ministral3")
GEMMA4_PATTERNS = ("gemma-4", "gemma4", "gemma-4-12b")


def _is_mistral3(model_name: str) -> bool:
    n = model_name.lower()
    return any(p in n for p in MISTRAL3_PATTERNS)

def _is_gemma4(model_name: str) -> bool:
    n = model_name.lower()
    return any(p in n for p in GEMMA4_PATTERNS)

# AirLLM import
try:
    from airllm import AutoModel as AirLLMAutoModel
    AIRLLM_AVAILABLE = True
except ImportError:
    AIRLLM_AVAILABLE = False
    # Don't print warning - it's optional


class LLMWrapper(ABC):
    """Abstract base class for LLM interactions"""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate response from prompt"""
        pass

    @abstractmethod
    def get_attention(self, prompt: str) -> Optional[torch.Tensor]:
        """Get attention weights (None for black-box models)"""
        pass

    @abstractmethod
    def get_logits(self, prompt: str) -> Optional[torch.Tensor]:
        """Get output logits (None for black-box models)"""
        pass

    @abstractmethod
    def get_hidden_states(self, prompt: str) -> Optional[List[torch.Tensor]]:
        """Get hidden states from all layers (None for black-box models)"""
        pass


class AirLLMWrapper(LLMWrapper):
    """
    Wrapper for AirLLM - enables running very large models on limited VRAM
    
    AirLLM uses layer-by-layer inference, loading only one layer at a time.
    This allows running 70B+ models on GPUs with 4-8GB VRAM.
    
    Limitations:
    - Slower inference (trades speed for memory)
    - No attention output support
    - No hidden states access
    - Generation only (no logits access during generation)
    
    Example:
        model = AirLLMWrapper("deepseek-ai/DeepSeek-R1", compression="4bit")
        response = model.generate("Hello!")
    """

    def __init__(
        self,
        model_name: str,
        compression: str = "4bit",  # "4bit", "8bit", or None
        max_seq_len: int = 2048,
        device: str = "cuda"
    ):
        if not AIRLLM_AVAILABLE:
            raise ImportError(
                "airllm not installed. Install with: pip install airllm\n"
                "For 4-bit support: pip install airllm bitsandbytes"
            )

        self.model_name = model_name
        self.device = device
        self.compression = compression
        self.max_seq_len = max_seq_len
        self.can_get_attention = False

        print(f"Loading {model_name} with AirLLM...")
        print(f"  Compression: {compression}")
        print(f"  Max sequence length: {max_seq_len}")

        # AirLLM configuration
        airllm_kwargs = {
            "max_seq_len": max_seq_len,
        }

        if compression == "4bit":
            airllm_kwargs["compression"] = "4bit"
            print("  Using 4-bit compression (minimal VRAM)")
        elif compression == "8bit":
            airllm_kwargs["compression"] = "8bit"
            print("  Using 8-bit compression")

        # Load model with AirLLM
        self.model = AirLLMAutoModel.from_pretrained(
            model_name,
            **airllm_kwargs
        )

        # Load tokenizer separately (AirLLM doesn't expose it directly)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"✓ Model loaded with AirLLM!")
        print(f"  Note: AirLLM uses layer-by-layer inference (slower but memory efficient)")
        print(f"  Attention/hidden states NOT available with AirLLM")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False
    ) -> str:
        """Generate response using AirLLM"""

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len - max_tokens
        )

        input_ids = inputs["input_ids"].to(self.device)

        # Generate with AirLLM
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
        }

        if do_sample and temperature > 0:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        # AirLLM generate
        output_ids = self.model.generate(
            input_ids,
            **generation_kwargs
        )

        # Decode only new tokens
        generated_ids = output_ids[0][input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response

    def get_attention(self, prompt: str) -> None:
        """Not available for AirLLM (layer-by-layer inference)"""
        warnings.warn("Attention not available with AirLLM (uses layer-by-layer inference)")
        return None

    def get_logits(self, prompt: str) -> None:
        """Not available for AirLLM"""
        warnings.warn("Logits not available with AirLLM")
        return None

    def get_hidden_states(self, prompt: str) -> None:
        """Not available for AirLLM"""
        warnings.warn("Hidden states not available with AirLLM")
        return None

    def tokenize(self, text: str, return_tensors: bool = False):
        """Tokenize text"""
        if return_tensors:
            return self.tokenizer(text, return_tensors="pt")
        else:
            tokens = self.tokenizer.tokenize(text)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            return tokens, token_ids


class HuggingFaceLLM(LLMWrapper):
    """
    Enhanced wrapper for HuggingFace models with full layer access
    Supports: Llama, Mistral, Gemma, Phi, Qwen, DeepSeek, etc.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype = torch.float16,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_flash_attention: bool = False,
        force_eager_attention: bool = True  # Force eager for attention access
    ):
        self.device = device
        self.model_name = model_name
        self.use_device_map = (device == "cuda")
        self.is_gemma4 = _is_gemma4(model_name)
        self.is_mistral3 = _is_mistral3(model_name)

        print(f"Loading {model_name}...")
        print(f"  Device: {device}")
        print(f"  Dtype: {torch_dtype}")

        # Configure model loading
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
        }

        # Only use device_map for CUDA (handles multi-GPU automatically)
        if device == "cuda":
            model_kwargs["device_map"] = "auto"

        # Check if model is already quantized
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

            if hasattr(config, 'quantization_config') and config.quantization_config is not None:
                quant_config = config.quantization_config
                if isinstance(quant_config, dict):
                    quant_type = quant_config.get('quant_method', 'Unknown')
                else:
                    quant_type = type(quant_config).__name__

                print(f"  ⚠ Model is already quantized with {quant_type}")
                print(f"    Ignoring load_in_8bit and load_in_4bit parameters")
                load_in_8bit = False
                load_in_4bit = False
        except Exception as e:
            pass

        # Quantization options (only if not already quantized)
        if load_in_8bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )

            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
            print("  Using 8-bit quantization (BitsAndBytesConfig)")

        elif load_in_4bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
            print("  Using 4-bit quantization (BitsAndBytesConfig)")

        # Attention implementation
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("  Using Flash Attention 2 (attention outputs not available)")
            self.can_get_attention = False
        elif force_eager_attention:
            model_kwargs["attn_implementation"] = "eager"
            print("  Using eager attention (enables attention output)")
            self.can_get_attention = True
        else:
            print("  Using default attention implementation")
            self.can_get_attention = False

                # Load tokenizer / processor and model
        if self.is_gemma4:
            from transformers import AutoProcessor, AutoModelForMultimodalLM
            print("  Gemma 4 detected -- loading via AutoModelForMultimodalLM")
            self.processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True
            )
            # The processor wraps a tokenizer; expose it so every existing
            # reference to self.tokenizer (terminators, padding, vocab size,
            # get_logits, get_hidden_states) keeps working unchanged.
            self.tokenizer = self.processor.tokenizer
            if torch_dtype == torch.float16:
                print("  Gemma 4 requires bfloat16 -- overriding torch_dtype")
            torch_dtype = torch.bfloat16
            model_kwargs["torch_dtype"] = torch_dtype
            self.model = AutoModelForMultimodalLM.from_pretrained(
                model_name, **model_kwargs
            )
        elif self.is_mistral3:
            from transformers import (MistralCommonBackend,
                                      Mistral3ForConditionalGeneration,
                                      FineGrainedFP8Config)
            print("  Ministral 3 detected -- MistralCommonBackend tokenizer, "
                  "Mistral3ForConditionalGeneration, FP8 dequantized to BF16")
            self.processor = None
            self.tokenizer = MistralCommonBackend.from_pretrained(model_name)
            model_kwargs.pop("torch_dtype", None)
            self.model = Mistral3ForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=FineGrainedFP8Config(dequantize=True),
                **model_kwargs,
            )
        else:
            self.processor = None
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, **model_kwargs
            )
        # Set to evaluation mode
        self.model.eval()

        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        # Move to device if not using device_map
        if device == "cpu":
            self.model = self.model.to(device)

        # Determine the actual device for the model
        if hasattr(self.model, 'hf_device_map'):
            self.model_device = list(self.model.hf_device_map.values())[0]
            print(f"  Model distributed across: {self.model.hf_device_map}")
        else:
            self.model_device = next(self.model.parameters()).device
            print(f"  Model on device: {self.model_device}")

        print(f"✓ Model loaded successfully!")
        print(f"  Vocabulary size: {len(self.tokenizer)}")
        print(f"  Model parameters: {self.model.num_parameters() / 1e9:.2f}B")
        print(f"  Attention outputs available: {self.can_get_attention}")

    def _prepare_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move inputs to the correct device"""
        if hasattr(self.model, 'hf_device_map'):
            target_device = self.model_device
        else:
            target_device = self.model_device

        return {k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()}

    def _get_terminators(self):
        """Collect every plausible end-of-turn token id for this model."""
        ids = set()
    
        if self.tokenizer.eos_token_id is not None:
            ids.add(self.tokenizer.eos_token_id)
    
        # Models add their turn terminator to the added-vocab, not to eos_token.
        added = set(self.tokenizer.get_vocab().keys())
        for tok in ("<|eot_id|>",        # Llama-3.x
                    "<|im_end|>",        # Qwen / ChatML
                    "<|end_of_text|>",   # Llama-3.x document end
                    "<|endoftext|>",
                    "</s>"):             # Llama-2
            if tok in added:
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                if isinstance(tid, int) and tid >= 0:
                    ids.add(tid)
    
        return sorted(ids)
    
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        use_chat_template: bool = True,
        system_prompt: str = None,
        enable_thinking: bool = False,
        return_meta: bool = False
    ):
        """
        Generate a response.
    
        Args:
            use_chat_template: Apply the model's instruct format. Must be True to
                evaluate a model in its deployed configuration. False reproduces
                the pre-patch raw-completion behaviour.
            system_prompt: Optional system turn. Leave None to match a bare chat
                interface, which is what the LIBRA threat model specifies.
            return_meta: If True, return (response, meta_dict) instead of a bare
                string, so the caller can record what was actually sent.
        """
        template_applied = False
        templated_text = prompt

        # Gemma 4 holds the chat template on the processor, not the tokenizer.
        tmpl_fn = None
        if use_chat_template:
            if self.is_gemma4 and self.processor is not None:
                tmpl_fn = self.processor.apply_chat_template
            elif getattr(self.tokenizer, "chat_template", None):
                tmpl_fn = self.tokenizer.apply_chat_template
            elif self.is_mistral3:
                # MistralCommonBackend builds the template in code rather than
                # exposing it as a string attribute, so chat_template is None
                # even though apply_chat_template works. Without this branch the
                # model would silently fall through to raw-completion mode.
                tmpl_fn = self.tokenizer.apply_chat_template

        if tmpl_fn is not None:
            messages = []
            if self.is_gemma4:
                # Thinking is triggered by a <|think|> token at the START of the
                # SYSTEM prompt -- not by an enable_thinking kwarg the way Qwen3
                # does it. No token means no thinking.
                sys_content = system_prompt or ""
                if enable_thinking:
                    sys_content = "<|think|>" + sys_content
                if sys_content:
                    messages.append({"role": "system", "content": sys_content})
            elif system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            try:
                try:
                    templated_text = tmpl_fn(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=enable_thinking,
                    )
                except TypeError:
                    # Llama/Mistral/Gemma templates reject enable_thinking.
                    templated_text = tmpl_fn(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                template_applied = True
            except Exception as e:
                warnings.warn(
                    f"apply_chat_template failed for {self.model_name}: {e}. "
                    f"Falling back to raw prompt -- results will NOT reflect the "
                    f"deployed configuration."
                )
                templated_text = prompt
    
        elif use_chat_template:
            warnings.warn(
                f"{self.model_name} has no chat_template defined. Generating in "
                f"raw-completion mode. If this is an instruct model, the tokenizer "
                f"is wrong."
            )
    
        # add_special_tokens=False: apply_chat_template already inserts BOS.
        # Adding it again produces a duplicated BOS and degrades output.
        inputs = self.tokenizer(
            templated_text,
            return_tensors="pt",
            add_special_tokens=not template_applied,
        )
        inputs = self._prepare_inputs(inputs)
    
        terminators = self._get_terminators()
    
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": terminators,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
    
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
    
        generated = outputs[0][inputs["input_ids"].shape[1]:]

        if self.is_gemma4:
            # Gemma 4 LOOPS on adversarial prompts: it answers, re-opens a
            # channel, and answers again until the token cap. That is why
            # truncation stayed at ~84% whether the cap was 512 or 1024 --
            # raising it just produces more copies, so take the FIRST completed
            # answer rather than the last.
            #
            # Decode WITH special tokens: skip_special_tokens=True removes the
            # angle brackets and leaves a bare "thought" in the text, which is
            # unsplittable. parse_response is not used -- it needs a prefix=
            # argument and, called without one, corrupts the output.
            raw = self.processor.decode(generated, skip_special_tokens=False)
            had_thinking = ("thought" in raw) or ("<|channel>" in raw)

            body = raw
            # 1. drop everything up to and including the FIRST closing marker
            for close in ("<channel|>", "</thought>"):
                if close in body:
                    body = body.partition(close)[2]
                    break
            # 2. cut at the next OPENING marker -- that is where the loop
            #    restarts, and everything after it is a duplicate
            for open_ in ("<|channel>", "<thought>"):
                if open_ in body:
                    body = body.partition(open_)[0]
                    break

            response = body
            for marker in ("<turn|>", "<|turn>", "<|channel>", "<channel|>",
                           "<thought>", "</thought>", "<bos>", "<eos>",
                           "<end_of_turn>"):
                response = response.replace(marker, "")
            # Any residual bare marker left by an unbracketed decode.
            response = response.replace("thought\n", "").replace("thought", "")

            # Gemma 4 degenerates after completing its answer, filling the rest
            # of the budget with repeated filler -- rows of underscores, literal
            # "/n" sequences, and similar. That is what drives the ~84%
            # truncation rate, not verbosity, which is why raising the cap from
            # 512 to 1024 changed nothing. Cut at the first long run so the
            # judges score the answer rather than the padding.
            m = re.search(r"(?:_{10,}|(?:/n\s*){5,})", response)
            if m:
                response = response[:m.start()]

            if not response.strip():
                # Nothing survived: the whole generation was reasoning, cut off
                # before any answer. Blank rather than let the judges score
                # reasoning as if it were the answer -- counting the cell as a
                # failure is the conservative direction.
                response = ""
        else:
            response = self.tokenizer.decode(generated, skip_special_tokens=True)
            # Qwen3 emits <think>...</think>. Strip it so the judges score the
            # answer, not the reasoning trace.
            had_thinking = "</think>" in response
            if had_thinking:
                response = response.rpartition("</think>")[2]

        response = response.strip()

        if return_meta:
            meta = {
                "chat_template_applied": template_applied,
                "system_prompt_used": system_prompt is not None,
                "enable_thinking": enable_thinking,
                "had_thinking_block": had_thinking,
                "n_prompt_tokens": int(inputs["input_ids"].shape[1]),
                "n_generated_tokens": int(generated.shape[0]),
                "hit_token_cap": int(generated.shape[0]) >= max_tokens,
                "terminators": terminators,
            }
            return response, meta

        return response

    def get_attention(
        self,
        prompt: str,
        return_all_layers: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor], None]:
        """
        Get attention weights from model

        Args:
            prompt: Input text
            return_all_layers: If True, return list of attention from all layers
                             If False, return only last layer

        Returns:
            Attention tensor(s) with shape [batch, heads, seq_len, seq_len]
            Returns None if attention is not available
        """
        if not self.can_get_attention:
            warnings.warn(
                "Attention outputs not available with current attention implementation. "
                "Initialize model with force_eager_attention=True or use_flash_attention=False"
            )
            return None

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        if outputs.attentions is None or len(outputs.attentions) == 0:
            warnings.warn("Model did not return attention outputs")
            return None

        if return_all_layers:
            return list(outputs.attentions)
        else:
            return outputs.attentions[-1]

    def get_attention_for_generation(
        self,
        prompt: str,
        max_tokens: int = 50
    ) -> Union[List[torch.Tensor], None]:
        """
        Get attention during generation (cross-attention)
        Returns attention for each generated token
        """
        if not self.can_get_attention:
            warnings.warn("Attention outputs not available with current implementation")
            return None

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                output_attentions=True,
                return_dict_in_generate=True,
                do_sample=False
            )

        if not hasattr(outputs, 'attentions') or outputs.attentions is None:
            warnings.warn("Model did not return attention during generation")
            return None

        return outputs.attentions

    def get_logits(self, prompt: str) -> torch.Tensor:
        """
        Get output logits

        Returns:
            Logits tensor with shape [batch, seq_len, vocab_size]
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.logits

    def get_hidden_states(
        self,
        prompt: str,
        layer: Optional[int] = None
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Get hidden states from transformer layers

        Args:
            prompt: Input text
            layer: Specific layer index (None = all layers)

        Returns:
            Hidden states tensor(s) with shape [batch, seq_len, hidden_dim]
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states

        if layer is not None:
            return hidden_states[layer]
        else:
            return list(hidden_states)

    def get_perplexity(self, text: str) -> float:
        """
        Calculate perplexity of text
        Lower perplexity = model is more confident
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs['input_ids'])

        return torch.exp(outputs.loss).item()

    def tokenize(self, text: str, return_tensors: bool = False):
        """Tokenize text and optionally return token strings"""
        if return_tensors:
            return self.tokenizer(text, return_tensors="pt")
        else:
            tokens = self.tokenizer.tokenize(text)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            return tokens, token_ids

    def get_attention_map(
        self,
        prompt: str,
        layer: int = -1,
        head: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Get attention map for visualization

        Args:
            prompt: Input text
            layer: Which layer to visualize (-1 = last layer)
            head: Which attention head (None = average over all heads)

        Returns:
            Attention matrix as numpy array [seq_len, seq_len]
            Returns None if attention is not available
        """
        attention = self.get_attention(prompt, return_all_layers=True)

        if attention is None:
            return None

        layer_attention = attention[layer]
        layer_attention = layer_attention[0]

        if head is not None:
            attention_matrix = layer_attention[head].cpu().numpy()
        else:
            attention_matrix = layer_attention.mean(dim=0).cpu().numpy()

        return attention_matrix

    def set_attention_implementation(self, implementation: str):
        """
        Change attention implementation dynamically

        Args:
            implementation: One of ['eager', 'sdpa', 'flash_attention_2']
        """
        if implementation == 'eager':
            self.model.set_attn_implementation('eager')
            self.can_get_attention = True
            print("Switched to eager attention (attention outputs enabled)")
        elif implementation == 'sdpa':
            self.model.set_attn_implementation('sdpa')
            self.can_get_attention = False
            print("Switched to SDPA attention (attention outputs disabled)")
        elif implementation == 'flash_attention_2':
            self.model.set_attn_implementation('flash_attention_2')
            self.can_get_attention = False
            print("Switched to Flash Attention 2 (attention outputs disabled)")
        else:
            raise ValueError(f"Unknown attention implementation: {implementation}")


class GPTWrapper(LLMWrapper):
    """Wrapper for OpenAI GPT models (GPT-3.5, GPT-4, GPT-5, etc.)"""

    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Run: pip install openai")

        self.model_name = model_name
        self.can_get_attention = False

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter"
            )

        self.client = openai.OpenAI(api_key=api_key)
        print(f"Initialized OpenAI client with model: {model_name}")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        """Generate response via OpenAI API"""
        import time

        is_gpt5 = self.model_name.startswith('gpt-5')
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                if is_gpt5:
                    response = self.client.responses.create(
                        model=self.model_name,
                        input=prompt,
                        safety_identifier="libra-research",
                    )
                    result = response.output_text
                else:
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[{"role": "user", "content": prompt}],
                            max_completion_tokens=max_tokens,
                            temperature=temperature
                        )
                        result = response.choices[0].message.content
                    except Exception as e:
                        if "max_completion_tokens" in str(e) or "Unsupported parameter" in str(e):
                            response = self.client.chat.completions.create(
                                model=self.model_name,
                                messages=[{"role": "user", "content": prompt}],
                                max_tokens=max_tokens,
                                temperature=temperature
                            )
                            result = response.choices[0].message.content
                        else:
                            raise

                if result and result.strip():
                    return result

                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise

        return f"[Empty response from {self.model_name}]"

    def get_attention(self, prompt: str) -> None:
        return None

    def get_logits(self, prompt: str) -> None:
        return None

    def get_hidden_states(self, prompt: str) -> None:
        return None


class ClaudeWrapper(LLMWrapper):
    """Wrapper for Anthropic Claude models"""

    def __init__(self, model_name: str = "claude-sonnet-4-5-20250929", api_key: Optional[str] = None):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        self.model_name = model_name
        self.can_get_attention = False

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter"
            )

        self.client = anthropic.Anthropic(api_key=api_key)
        print(f"Initialized Anthropic client with model: {model_name}")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        """Generate response via Anthropic API"""
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def get_attention(self, prompt: str) -> None:
        return None

    def get_logits(self, prompt: str) -> None:
        return None

    def get_hidden_states(self, prompt: str) -> None:
        return None


class GeminiWrapper(LLMWrapper):
    """Wrapper for Google Gemini models (updated API)"""

    def __init__(self, model_name: str = "gemini-2.0-flash-exp", api_key: Optional[str] = None):
        if not GOOGLE_AVAILABLE:
            raise ImportError("google-genai package not installed. Run: pip install google-genai")

        self.model_name = model_name
        self.can_get_attention = False

        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter"
            )

        os.environ["GEMINI_API_KEY"] = api_key
        self.client = genai.Client()
        print(f"Initialized Gemini client with model: {model_name}")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        """Generate response via Gemini API"""
        config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )

        return response.text

    def get_attention(self, prompt: str) -> None:
        return None

    def get_logits(self, prompt: str) -> None:
        return None

    def get_hidden_states(self, prompt: str) -> None:
        return None


class XAIWrapper(LLMWrapper):
    """Wrapper for xAI Grok models"""

    def __init__(self, model_name: str = "grok-4-0709", api_key: Optional[str] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed (xAI uses OpenAI-compatible API)")

        self.model_name = model_name
        self.can_get_attention = False

        api_key = api_key or os.environ.get("XAI_API_KEY")
        if not api_key:
            raise ValueError(
                "xAI API key not found. Set XAI_API_KEY environment variable "
                "or pass api_key parameter"
            )

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
        print(f"Initialized xAI client with model: {model_name}")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        """Generate response via xAI API"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content

    def get_attention(self, prompt: str) -> None:
        return None

    def get_logits(self, prompt: str) -> None:
        return None

    def get_hidden_states(self, prompt: str) -> None:
        return None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def estimate_model_size(model_name: str) -> float:
    """
    Estimate model size in billions of parameters from name
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        Estimated size in billions
    """
    name_lower = model_name.lower()

    size_patterns = {
        '671b': 671, '405b': 405, '120b': 120, '70b': 70, '65b': 65,
        '34b': 34, '32b': 32, '24b': 24, '20b': 20, '14b': 14,
        '8b': 8, '7b': 7, '4b': 4, '3b': 3, '2b': 2, '1b': 1,
    }

    for pattern, size in size_patterns.items():
        if pattern in name_lower:
            return size

    # Special cases for known large models
    if 'deepseek-r1' in name_lower and 'distill' not in name_lower:
        return 671
    if 'deepseek-v3' in name_lower:
        return 671
    if 'mistral-large' in name_lower:
        return 123

    return 7  # Default


def get_gpu_memory_gb() -> float:
    """Get available GPU memory in GB"""
    if not torch.cuda.is_available():
        return 0

    try:
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        return total - reserved
    except:
        return 0


def estimate_memory_requirements(model_name: str) -> Dict[str, float]:
    """
    Estimate memory requirements for different precision levels
    
    Returns:
        Dictionary with memory estimates in GB
    """
    model_size = estimate_model_size(model_name)
    overhead = 1.2  # 20% overhead for activations etc.
    
    return {
        'model_size_b': model_size,
        'fp16': model_size * 2 * overhead,
        'int8': model_size * 1 * overhead,
        'int4': model_size * 0.5 * overhead,
    }


def should_use_airllm(model_name: str, available_memory_gb: float) -> bool:
    """
    Determine if AirLLM should be used for this model
    
    Args:
        model_name: HuggingFace model name
        available_memory_gb: Available GPU memory in GB
        
    Returns:
        True if AirLLM is recommended
    """
    model_size = estimate_model_size(model_name)
    int4_memory = model_size * 0.5 * 1.2

    # Use AirLLM if:
    # 1. Model doesn't fit even with 4-bit quantization
    # 2. Model is very large (>30B parameters)
    if model_size > 30 or int4_memory > available_memory_gb:
        return True

    return False


def determine_loading_strategy(model_name: str, available_memory_gb: float) -> Dict:
    """
    Determine optimal loading strategy based on available memory
    
    Args:
        model_name: HuggingFace model name
        available_memory_gb: Available GPU memory in GB
        
    Returns:
        Dictionary with loading configuration
    """
    mem_req = estimate_memory_requirements(model_name)
    
    if available_memory_gb >= mem_req['fp16']:
        return {
            'method': 'standard',
            'quantization': None,
            'reason': f"Fits in FP16 ({mem_req['fp16']:.0f}GB needed)"
        }
    elif available_memory_gb >= mem_req['int8']:
        return {
            'method': 'standard',
            'quantization': '8bit',
            'reason': f"Fits with INT8 ({mem_req['int8']:.0f}GB needed)"
        }
    elif available_memory_gb >= mem_req['int4']:
        return {
            'method': 'standard',
            'quantization': '4bit',
            'reason': f"Fits with INT4 ({mem_req['int4']:.0f}GB needed)"
        }
    else:
        return {
            'method': 'airllm',
            'quantization': '4bit',
            'reason': f"Too large ({mem_req['int4']:.0f}GB needed), using AirLLM"
        }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def load_model(
    model_type: str,
    model_name: str,
    device: str = "cuda",
    use_airllm: Optional[bool] = None,
    airllm_compression: str = "4bit",
    **kwargs
) -> LLMWrapper:
    """
    Factory function to load appropriate model wrapper
    
    Args:
        model_type: One of ["huggingface", "airllm", "gpt", "claude", "gemini", "xai"]
        model_name: Specific model identifier
        device: Device for local models ("cuda" or "cpu")
        use_airllm: Force AirLLM (None = auto-detect based on memory)
        airllm_compression: "4bit", "8bit", or None (for AirLLM)
        **kwargs: Additional arguments passed to wrapper
            For HuggingFace models:
                - force_eager_attention: bool (default True) - Enable attention outputs
                - use_flash_attention: bool (default False) - Use Flash Attention 2
                - load_in_8bit: bool (default False) - 8-bit quantization
                - load_in_4bit: bool (default False) - 4-bit quantization
    
    Returns:
        Initialized LLMWrapper instance
    
    Examples:
        # Load local HuggingFace model with attention access
        model = load_model("huggingface", "meta-llama/Llama-2-7b-chat-hf")
        
        # Load with quantization
        model = load_model("huggingface", "meta-llama/Llama-2-7b-chat-hf",
                          load_in_8bit=True)
        
        # Load large model with AirLLM (auto-detected)
        model = load_model("huggingface", "deepseek-ai/DeepSeek-R1")
        
        # Force AirLLM for any model
        model = load_model("huggingface", "meta-llama/Llama-2-7b-chat-hf",
                          use_airllm=True)
        
        # Load GPT-4
        model = load_model("gpt", "gpt-4")
        
        # Load Claude
        model = load_model("claude", "claude-3-5-sonnet-20241022")
        
        # Load Gemini
        model = load_model("gemini", "gemini-2.0-flash-exp")
        
        # Load Grok
        model = load_model("xai", "grok-4-0709")
    """
    model_type = model_type.lower()

    # API models
    if model_type in ["gpt", "openai"]:
        return GPTWrapper(model_name, **kwargs)
    elif model_type in ["claude", "anthropic"]:
        return ClaudeWrapper(model_name, **kwargs)
    elif model_type in ["gemini", "google"]:
        return GeminiWrapper(model_name, **kwargs)
    elif model_type in ["xai", "grok"]:
        return XAIWrapper(model_name, **kwargs)

    # Local models
    elif model_type in ["huggingface", "hf", "local"]:
        available_memory = get_gpu_memory_gb()

        # Auto-detect if should use AirLLM
        if use_airllm is None:
            use_airllm = should_use_airllm(model_name, available_memory)
            if use_airllm:
                print(f"⚠ Model too large for standard loading ({estimate_model_size(model_name)}B params)")
                print(f"  Using AirLLM for layer-by-layer inference")

        if use_airllm:
            if not AIRLLM_AVAILABLE:
                raise ImportError(
                    "AirLLM required for this model but not installed.\n"
                    "Install with: pip install airllm"
                )
            return AirLLMWrapper(
                model_name,
                compression=airllm_compression,
                device=device
            )
        else:
            return HuggingFaceLLM(model_name, device=device, **kwargs)

    # Explicit AirLLM
    elif model_type == "airllm":
        if not AIRLLM_AVAILABLE:
            raise ImportError("AirLLM not installed. Run: pip install airllm")
        return AirLLMWrapper(model_name, compression=airllm_compression, device=device)

    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Choose from: huggingface, airllm, gpt, claude, gemini, xai"
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_with_airllm(model_name: str, compression: str = "4bit", **kwargs) -> AirLLMWrapper:
    """Force load any model with AirLLM"""
    return load_model("airllm", model_name, airllm_compression=compression, **kwargs)


def load_llama(model_size: str = "7b", version: str = "2", chat: bool = True, device: str = "cuda", **kwargs):
    """Load Llama model"""
    if version == "2":
        variant = "chat" if chat else ""
        model_name = f"meta-llama/Llama-2-{model_size}-{variant}-hf"
    elif version == "3":
        model_name = f"meta-llama/Llama-3-{model_size}-hf"
    elif version == "3.1":
        model_name = f"meta-llama/Llama-3.1-{model_size}-hf"
    elif version == "3.2":
        model_name = f"meta-llama/Llama-3.2-{model_size.upper()}"
    else:
        raise ValueError(f"Unknown Llama version: {version}")

    return load_model("huggingface", model_name, device=device, **kwargs)


def load_mistral(variant: str = "7B-Instruct-v0.3", device: str = "cuda", **kwargs):
    """Load Mistral model"""
    model_name = f"mistralai/Mistral-{variant}"
    return load_model("huggingface", model_name, device=device, **kwargs)


def load_gemma(model_size: str = "7b", version: str = "2", instruct: bool = True, device: str = "cuda", **kwargs):
    """Load Gemma model"""
    variant = "it" if instruct else ""
    model_name = f"google/gemma-{version}-{model_size}-{variant}"
    return load_model("huggingface", model_name, device=device, **kwargs)


def load_qwen(model_size: str = "4B", variant: str = "Qwen3Guard-Gen", device: str = "cuda", **kwargs):
    """Load Qwen model"""
    model_name = f"Qwen/{variant}-{model_size}"
    return load_model("huggingface", model_name, device=device, **kwargs)


def load_deepseek(model_name: str = "DeepSeek-R1", device: str = "cuda", **kwargs):
    """Load DeepSeek model (auto-selects AirLLM for large models)"""
    full_name = f"deepseek-ai/{model_name}"
    return load_model("huggingface", full_name, device=device, **kwargs)


# ============================================================================
# MAIN - Testing
# ============================================================================

if __name__ == "__main__":
    """Test the model wrappers"""

    print("="*60)
    print("Model Wrapper V2 - Testing")
    print("="*60)

    # Check available backends
    print(f"\nAvailable backends:")
    print(f"  Transformers: Available")
    print(f"  AirLLM: {AIRLLM_AVAILABLE}")
    print(f"  OpenAI: {OPENAI_AVAILABLE}")
    print(f"  Anthropic: {ANTHROPIC_AVAILABLE}")
    print(f"  Google: {GOOGLE_AVAILABLE}")

    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {get_gpu_memory_gb():.1f} GB available")

    # Test model size estimation
    test_models = [
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-3.1-8B-Instruct",
        "deepseek-ai/DeepSeek-R1",
        "mistralai/Mistral-Large-Instruct-2411",
        "Qwen/Qwen3Guard-Gen-4B",
    ]

    print(f"\nModel analysis:")
    available_gb = get_gpu_memory_gb()
    
    for model in test_models:
        size = estimate_model_size(model)
        strategy = determine_loading_strategy(model, available_gb)
        print(f"\n  {model}")
        print(f"    Size: ~{size}B params")
        print(f"    Strategy: {strategy['method']} - {strategy['reason']}")

    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)