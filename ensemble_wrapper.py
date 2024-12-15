import textattack
import random
import torch
import transformers

from textattack.models.wrappers import HuggingFaceModelWrapper

class EnsembleModel(HuggingFaceModelWrapper):
    def __init__(self, model, tokenizer, ensemble_method='logit', num_samples=10, mask_percentage=5):
        super().__init__(model, tokenizer)
        self.ensemble_method = ensemble_method
        self.num_samples = num_samples
        self.mask_percentage = mask_percentage

    def _random_mask(self, text_list):
        """
        For each original text, create `num_samples` perturbed versions by randomly masking 
        a certain percentage of the words. Returns the perturbed texts and a list 
        mapping each perturbed text back to its original index.
        """
        all_perturbed_texts = []
        original_indices = []
        
        for i, text in enumerate(text_list):
            words = text.split()
            num_to_mask = max(1, int(len(words) * self.mask_percentage / 100))
            for _ in range(self.num_samples):
                mask_indices = random.sample(range(len(words)), num_to_mask)
                perturbed_words = words[:]
                for idx in mask_indices:
                    perturbed_words[idx] = '[MASK]'
                perturbed_text = ' '.join(perturbed_words)
                all_perturbed_texts.append(perturbed_text)
                original_indices.append(i)
        
        return all_perturbed_texts, original_indices

    def _ensemble(self, logits, original_indices):
        """
        Given a single batched logits tensor for all perturbed inputs, group them by 
        their original index and apply the ensemble method.
        
        logits: (N * num_samples, num_labels)
        original_indices: list of length (N * num_samples)
        Returns: (N, num_labels)
        """
        original_indices = torch.tensor(original_indices, dtype=torch.long)
        unique_indices = torch.unique(original_indices)

        ensembled_logits_list = []
        for idx in unique_indices:
            group_mask = (original_indices == idx)
            group_logits = logits[group_mask]  # shape: (num_samples, num_labels)

            if self.ensemble_method == 'logit':
                # Average the logits
                ens_logits = group_logits.mean(dim=0, keepdim=True)  # (1, num_labels)
            elif self.ensemble_method == 'vote':
                # Compute majority vote of predictions
                preds = torch.argmax(group_logits, dim=1)
                # Majority vote
                voted_label = torch.mode(preds).values.item()
                # Convert voted_label into logit form:
                # One simple approach: create a one-hot logit vector:
                num_labels = group_logits.size(1)
                ens_logits = torch.full((1, num_labels), -float('inf'), device=group_logits.device)
                ens_logits[0, voted_label] = 0.0
            else:
                raise ValueError("Invalid ensemble method")

            ensembled_logits_list.append(ens_logits)

        return torch.cat(ensembled_logits_list, dim=0)  # (N, num_labels)

    def __call__(self, text_input_list):
        # text_input_list is a list of strings per TextAttack conventions
        if not all(isinstance(t, str) for t in text_input_list):
            # If text_input_list is a list of textattack.shared.AttackedText, extract text
            text_input_list = [t.text if hasattr(t, 'text') else str(t) for t in text_input_list]

        # Generate perturbed texts
        perturbed_text_list, original_indices = self._random_mask(text_input_list)

        # Get logits for all perturbed texts by calling the parent's __call__ method
        # This handles tokenization and model inference automatically.
        perturbed_logits = super().__call__(perturbed_text_list)
        # perturbed_logits: shape (len(perturbed_text_list), num_labels)

        # Now apply ensemble on a per-original-input basis
        ensembled_logits = self._ensemble(perturbed_logits, original_indices)

        return ensembled_logits
