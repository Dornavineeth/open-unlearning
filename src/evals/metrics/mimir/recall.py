"""
    ReCaLL Attack: https://github.com/ruoyuxie/recall/
"""
import torch 
import numpy as np
from evals.metrics.mimir.all_attacks import Attack
from evals.metrics.utils import evaluate_probability, extract_target_texts_from_processed_data

class ReCaLLAttack(Attack):

    #** Note: this is a suboptimal implementation of the ReCaLL attack due to necessary changes made to integrate it alongside the other attacks
    #** for a better performing version, please refer to: https://github.com/ruoyuxie/recall 
    
    def __init__(self, model, recall_dict):
        super().__init__(model)
        self.recall_dict = recall_dict
        self.prefix = None

    def preprocess_batch(self, tokens):
        """Get evaluation results and extracted text."""
        eval_results = evaluate_probability(self.model, tokens)
        texts = extract_target_texts_from_processed_data(self.model.tokenizer, tokens)
        return {
            'eval_results': eval_results,
            'texts': texts
        }
        
    def score_sample(self, sample_values):
        """Compute ReCaLL score for single sample."""
        loss = sample_values['eval_results'][0]['avg_loss']
        text = sample_values['texts'][0]
        
        ll_nonmember = self.get_conditional_ll(
            nonmember_prefix=self.recall_dict["prefix"],
            text=text,
            num_shots=self.recall_dict["num_shots"],
            avg_length=self.recall_dict["avg_length"]
        )
        return ll_nonmember / loss

    def process_prefix(self, prefix, avg_length, total_shots):
        model = self.target_model
        tokenizer = self.target_model.tokenizer

        if self.prefix is not None:
            # We only need to process the prefix once, after that we can just return
            return self.prefix

        max_length = model.max_length
        token_counts = [len(tokenizer.encode(shot)) for shot in prefix]

        target_token_count = avg_length
        total_tokens = sum(token_counts) + target_token_count
        if total_tokens<=max_length:
            self.prefix = prefix
            return self.prefix
        # Determine the maximum number of shots that can fit within the max_length
        max_shots = 0
        cumulative_tokens = target_token_count
        for count in token_counts:
            if cumulative_tokens + count <= max_length:
                max_shots += 1
                cumulative_tokens += count
            else:
                break
        # Truncate the prefix to include only the maximum number of shots
        truncated_prefix = prefix[-max_shots:]
        print(f"""\nToo many shots used. Initial ReCaLL number of shots was {total_shots}. Maximum number of shots is {max_shots}. Defaulting to maximum number of shots.""")
        self.prefix = truncated_prefix
        return self.prefix
    
    def get_conditional_ll(self, nonmember_prefix, text, num_shots, avg_length):
        assert nonmember_prefix, "nonmember_prefix should not be None or empty"

        model = self.target_model
        tokenizer = self.target_model.tokenizer

        target_encodings = tokenizer(text=text, return_tensors="pt")

        processed_prefix = self.process_prefix(nonmember_prefix, avg_length, total_shots=num_shots)
        input_encodings = tokenizer(text="".join(processed_prefix), return_tensors="pt")

        prefix_ids = input_encodings.input_ids.to(model.device)
        text_ids = target_encodings.input_ids.to(model.device)

        max_length = model.max_length

        if prefix_ids.size(1) >= max_length:
            raise ValueError("Prefix length exceeds or equals the model's maximum context window.")

        labels = torch.cat((prefix_ids, text_ids), dim=1)
        total_length = labels.size(1)

        total_loss = 0
        total_tokens = 0
        with torch.no_grad():
            for i in range(0, total_length, max_length):
                begin_loc = i
                end_loc = min(i + max_length, total_length)
                trg_len = end_loc - begin_loc
                
                input_ids = labels[:, begin_loc:end_loc].to(model.device)
                target_ids = input_ids.clone()
                
                if begin_loc < prefix_ids.size(1):
                    prefix_overlap = min(prefix_ids.size(1) - begin_loc, max_length)
                    target_ids[:, :prefix_overlap] = -100
                
                if end_loc > total_length - text_ids.size(1):
                    target_overlap = min(end_loc - (total_length - text_ids.size(1)), max_length)
                    target_ids[:, -target_overlap:] = input_ids[:, -target_overlap:]
                
                if torch.all(target_ids == -100):
                    continue
                
                outputs = model.model(input_ids, labels=target_ids)
                loss = outputs.loss
                if torch.isnan(loss):
                    print(f"NaN detected in loss at iteration {i}. Non masked target_ids size is {(target_ids != -100).sum().item()}")
                    continue
                non_masked_tokens = (target_ids != -100).sum().item()
                total_loss += loss.item() * non_masked_tokens
                total_tokens += non_masked_tokens

        average_loss = total_loss / total_tokens if total_tokens > 0 else 0
        return -average_loss



