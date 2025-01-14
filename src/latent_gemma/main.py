import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TextStreamer
from transformers import GemmaForCausalLM, DynamicCache, PreTrainedTokenizer, PreTrainedModel
from typing import Optional, List, Union, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class LatentReasoningGemmaForCausalLM(GemmaForCausalLM):
    """
    A custom implementation of GemmaForCausalLM that supports latent reasoning 
    using the Coconut (Chain of Continuous Thought) paradigm.
    """

    DEFAULT_CONFIG = {
        # Core Learning Parameters
        "continuous_thoughts": 4,               # Number of latent space reasoning steps
        "stages": 4,                            # Number of training curriculum stages
        "training_thoughts_sequence_length": 50, # Number of thought sequence to generate

        # Inference and Evaluation Params       
        "fuzzy_matcher_threshold": 80,          # Fuzzy matcher threshold at 80%
        "cot_decoding_k": 5,                    # Number of paths to try before finding the best answer

        # Model Setup
        "max_length": 256,                      # Maximum text length to process

        # Special Tokens
        "bot_id": "<bot>",                      # Marks start of latent reasoning
        "eot_id": "<eot>",                      # Marks end of latent reasoning
        "answer_id": "<answer>",                # Marks the begining of answer
        "debug": True,                          # Enables debugging output. Also allows you see the model's thoughts

    }
    
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer: PreTrainedTokenizer = None
        self.current_stage = 0
        self.model_config = type(self).DEFAULT_CONFIG
        self.debug = self.model_config.get("debug", False)
        self.diversity_weight = self.model_config.get("diversity_weight", 0.1)
        self.coherence_weight = self.model_config.get("coherence_weight", 0.1)

    def get_input_ids(self, inputs_embeds):
        """Helper method to get input ids from embeddings."""
        embedding_matrix = self.get_input_embeddings().weight
        similarities = torch.matmul(inputs_embeds, embedding_matrix.T)
        token_ids = torch.argmax(similarities, dim=-1)
        return token_ids

    def thoughts_forward(self, num_thoughts, thought_ids, thought_mask, num_of_thought_tokens = 1):
        """
        Generate continuous thought embeddings.
        """
        all_thought_outputs = []
        batch_size = thought_ids.shape[0]
        
        # Get initial embeddings
        initial_embeds = self.get_input_embeddings()(thought_ids)
        current_embeds = initial_embeds
        current_mask = thought_mask

        for t in range(num_thoughts):
            # Forward pass through transformer
            outputs = self.model.forward(
                inputs_embeds=current_embeds,
                attention_mask=current_mask,
                past_key_values=None,
                use_cache=False,
                return_dict=True,
                output_hidden_states=True,  # Get hidden states from all layers
            )
            
            # Get hidden states from all layers for better representation
            hidden_states = outputs.hidden_states
            
            # Combine hidden states from different layers using attention
            layer_attention = torch.softmax(
                torch.randn(len(hidden_states), device=hidden_states[0].device), 
                dim=0
            )
            weighted_states = sum(w * h for w, h in zip(layer_attention, hidden_states))
            
            n = num_of_thought_tokens
            last_hidden = weighted_states[:, -n:, :]  # [batch_size, n, hidden_size]
            
            # Project to lower dimension for thought space
            thought_proj = nn.Sequential(
                nn.Linear(last_hidden.shape[-1], self.config.hidden_size // 2),
                nn.LayerNorm(self.config.hidden_size // 2),
                nn.GELU()
            ).to(last_hidden.device)
            projected_thought = thought_proj(last_hidden)  # [batch_size, n, hidden_size // 2]
            
            # Add noise to increase diversity
            noise = torch.randn_like(projected_thought) * 0.1  # Adjust noise scale as needed
            projected_thought = projected_thought + noise
            
            # Project back to embedding space
            embed_proj = nn.Linear(
                self.config.hidden_size // 2,
                self.config.hidden_size,
                device=projected_thought.device
            )
            next_token_embeds = embed_proj(projected_thought)  # [batch_size, n, hidden_size]
            
            # Apply layer normalization for stability
            next_token_embeds = nn.LayerNorm(
                self.config.hidden_size,
                device=next_token_embeds.device
            )(next_token_embeds)
            
            # Update embeddings and mask
            current_embeds = torch.cat([current_embeds, next_token_embeds], dim=1)
            current_mask = torch.cat([
                current_mask,
                torch.ones((batch_size, n), device=current_mask.device)
            ], dim=1)
            
            all_thought_outputs.append(last_hidden)

        # Ensure reasonable sequence length
        max_seq_len = self.model_config.get("max_length", 512)
        if current_embeds.shape[1] > max_seq_len:
            current_embeds = current_embeds[:, :max_seq_len, :]
            current_mask = current_mask[:, :max_seq_len]
        
        return all_thought_outputs, current_embeds, current_mask


    def train_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ):
        """
        Training forward pass with continuous thought generation and CoT alignment.
        """
        self.train()

        # Keep original labels if none provided
        if labels is None:
            labels = input_ids.clone()
            batch_size = labels.shape[0]
            eot_id = self.tokenizer.convert_tokens_to_ids(self.model_config["eot_id"])

            for i in range(batch_size):
                # Find the positions of <eot> in the input_ids
                eot_pos = (input_ids[i] == eot_id).nonzero(as_tuple=True)

                if len(eot_pos[0]) > 0:
                    # Get the last occurrence of <eot>
                    last_eot_pos = eot_pos[0][-1].item()
                    
                    # Mask everything before and including the last <eot>
                    labels[i, :last_eot_pos] = -100

                # Mask padding
                labels[i, attention_mask[i] == 0] = -100

        # Get input embeddings if not provided
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Generate continuous thoughts
        if self.current_stage > 0:
            num_thoughts = self.current_stage * self.model_config["continuous_thoughts"]
            all_thoughts, final_embeds, final_mask = self.thoughts_forward(
                num_thoughts=num_thoughts,
                thought_ids=input_ids,
                thought_mask=attention_mask,
                num_of_thought_tokens = self.model_config["training_thoughts_sequence_length"]
            )

            # Add auxiliary losses
            auxiliary_losses = []

            # Thought coherence loss
            if len(all_thoughts) > 1:
                coherence_loss = 0
                for t1, t2 in zip(all_thoughts[:-1], all_thoughts[1:]):
                    sim = F.cosine_similarity(t1, t2, dim=-1)
                    coherence_loss += (1 - sim).mean()
                auxiliary_losses.append(coherence_loss * self.coherence_weight)

            batch_size = labels.shape[0]

            for i in range(batch_size):
                # Find the start and end of CoT in the labels
                cot_start = None
                
                for j, token_id in enumerate(labels[i]):
                    if token_id == self.tokenizer.convert_tokens_to_ids(self.model_config["eot_id"]):
                        cot_start = j + 1  # Start of CoT


                # Debugging: Print CoT tokens and latent thoughts
                if cot_start is not None:
                    # Extract CoT tokens
                    cot_tokens = labels[i, cot_start:]  # [cot_seq_len]

                    # Get the latent thoughts for this batch
                    latent_thoughts = all_thoughts[i]  # [thought_seq_len, hidden_size]

                    # Project latent thoughts to logits
                    thought_logits = self.lm_head(latent_thoughts)  # [thought_seq_len, vocab_size]
                    thought_token_ids = torch.argmax(thought_logits, dim=-1)  # [thought_seq_len]


                    # Debugging: Print CoT tokens and latent thoughts
                    if self.debug:
                        # Decode CoT tokens
                        cot_tokens_list = cot_tokens.squeeze().tolist()  # Convert to 1D list
                        if isinstance(cot_tokens_list, int):  # Handle single token case
                            cot_tokens_list = [cot_tokens_list]
                        cot_text = self.tokenizer.decode(cot_tokens_list, skip_special_tokens=True)
                        print(f" ==================== \n Debug: CoT for batch {i}: {cot_text} \n ====================")

                        # Decode latent thoughts
                        thought_token_ids_list = thought_token_ids.squeeze().tolist()  # Convert to list

                        # Ensure thought_token_ids_list is a flat list
                        if isinstance(thought_token_ids_list, list) and all(isinstance(item, list) for item in thought_token_ids_list):
                            # Flatten the nested list
                            thought_token_ids_list = [token for sublist in thought_token_ids_list for token in sublist]
                        elif isinstance(thought_token_ids_list, int):  # Handle single token case
                            thought_token_ids_list = [thought_token_ids_list]

                        # Decode the flat list of token IDs
                        thought_text = self.tokenizer.decode(thought_token_ids_list, skip_special_tokens=False)
                        print(f"==================== \n Debug: Latent thoughts for batch {i}: {thought_text} \n ========================")


            # Forward pass with thoughts
            outputs = super().forward(
                inputs_embeds=final_embeds,
                attention_mask=final_mask,
                labels=labels,
                **kwargs
            )

            # Add auxiliary losses
            if auxiliary_losses:
                outputs.loss += sum(auxiliary_losses)

        else:

            if inputs_embeds is None:
                # Standard forward pass for initial stage
                outputs = super().forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )
            else:

                outputs = super().forward(
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

        return outputs

    
    def infer_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[DynamicCache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ):
        """
        Inference forward pass with continuous thought generation.
        """

        batch_size = input_ids.shape[0]

        # Insert <bot> token to initiate latent reasoning
        if input_ids.shape[1] > 1:
            input_ids = torch.cat(
                [
                    input_ids,
                    torch.tensor(
                        [[self.tokenizer.convert_tokens_to_ids(self.model_config["bot_id"])]] * batch_size,
                        device=input_ids.device,
                    ),
                ],
                dim=1,
            )
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((batch_size, 1), device=attention_mask.device),
                ],
                dim=1,
            )

        # Generate continuous thoughts
        if self.model_config["stages"] - 1 > 0 and input_ids.shape[1] > 1:
            num_thoughts = (self.model_config["stages"] - 1) * self.model_config["continuous_thoughts"]
            all_thoughts, final_embeds, final_mask = self.thoughts_forward(
                num_thoughts, input_ids, attention_mask
            )

            # Add <eot> token to mark the end of latent reasoning
            eot_embeds = self.get_input_embeddings()(
                torch.tensor(
                    [[self.tokenizer.convert_tokens_to_ids(self.model_config["eot_id"])]] * batch_size,
                    device=final_embeds.device,
                )
            )
            final_embeds = torch.cat([final_embeds, eot_embeds], dim=1)
            final_mask = torch.cat([final_mask, torch.ones((batch_size, 1), device=final_mask.device)], dim=1)

            # Generate final output in language mode
            outputs = super().forward(
                inputs_embeds=final_embeds,
                attention_mask=final_mask,
                past_key_values=None,  # Reset past_key_values for answer generation
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
        else:
            # Standard forward pass (no latent thoughts)
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        return outputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[DynamicCache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ):
        """Main forward function that routes to either training or inference."""
        forward_fn = self.train_forward if self.training else self.infer_forward
        return forward_fn(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )
    
        
    def generate_answer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        question: str,
        max_length: Optional[int] = None,
        k: int = DEFAULT_CONFIG["cot_decoding_k"],
        temperature: float = 1.0,
        max_new_tokens: Optional[int] = None,
        **generation_kwargs
    ) -> str:
        """
        Generates answer using CoT decoding and returns the best path.
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            question: Input question
            max_length: Maximum sequence length
            k: Number of alternative paths to consider
            temperature: Sampling temperature
            **generation_kwargs: Additional generation arguments
            
        Returns:
            Best decoded sequence with highest confidence
        """
        # Initialize streamer
        streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=False)

        # Tokenize input
        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Get initial logits for CoT paths
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        first_token_logits = outputs.logits[:, -1, :] / temperature
        
        # Get top-k tokens
        probs = F.softmax(first_token_logits, dim=-1)
        top_k_probs, top_k_tokens = torch.topk(probs, k, dim=-1)
        
        best_path = None
        best_confidence = -float('inf')
        
        # Generate continuation for each top-k token
        for i in range(k):
            # Prepare input with current top-k token
            curr_input_ids = torch.cat([
                input_ids,
                top_k_tokens[:, i:i+1]
            ], dim=1)
            
            curr_attention_mask = torch.cat([
                attention_mask,
                torch.ones((attention_mask.shape[0], 1), device=model.device)
            ], dim=1)
            
            # Generate with streamer for best path
            if max_new_tokens is not None:
                outputs = model.generate(
                    input_ids=curr_input_ids,
                    attention_mask=curr_attention_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_length=None,
                    output_scores=True,
                    return_dict_in_generate=True,
                    streamer=streamer if i == 0 else None,  # Only stream first path
                    **generation_kwargs
                )
            else:
                outputs = model.generate(
                    input_ids=curr_input_ids,
                    attention_mask=curr_attention_mask,
                    max_length=max_length,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                    **generation_kwargs
                )
            
            # Calculate confidence for this path
            _, confidence = self.calculate_answer_confidence(
                outputs.sequences[0].tolist(),
                outputs.scores[-1],
                tokenizer
            )
            
            # Update best path if confidence is higher
            if confidence > best_confidence:
                best_confidence = confidence
                best_path = outputs.sequences[0]
                
        # Return the path with highest confidence
        return tokenizer.decode(best_path, skip_special_tokens=True)

    def calculate_answer_confidence(
        self,
        sequence: List[int],
        final_logits: torch.Tensor,
        tokenizer: PreTrainedTokenizer
    ) -> Tuple[str, float]:
        """Calculate confidence score using min-margin approach."""
        # Extract answer from sequence
        answer = self.extract_answer(sequence, tokenizer)
        
        if not answer:
            return "", 0.0
        
        # Get probabilities
        probs = F.softmax(final_logits, dim=-1)
        
        # Calculate margins for answer tokens
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
        margins = []
        
        for token in answer_tokens:
            token_prob = probs[0, token].item()
            sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
            second_best_prob = sorted_probs[0, 1].item()
            margin = token_prob - second_best_prob
            margins.append(margin)
            
        confidence = sum(margins) / len(margins)
        return answer, confidence

    def extract_answer(self, sequence: List[int], tokenizer: PreTrainedTokenizer) -> str:
        """
        Extract final answer from sequence using <eot> token.
        Finds the answer between the last occurrence of <eot> and the end of sequence.
        """
        # Convert sequence to string
        decoded = tokenizer.decode(sequence)
        
        # Find last <eot> position
        eot_position = decoded.rfind(LatentReasoningGemmaForCausalLM.DEFAULT_CONFIG["eot_id"])
        
        if eot_position != -1:
            # Extract everything after the last <eot>
            answer = decoded[eot_position + len(LatentReasoningGemmaForCausalLM.DEFAULT_CONFIG["eot_id"]):].strip()
            return answer
            
        return decoded