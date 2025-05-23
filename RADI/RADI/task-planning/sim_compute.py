import os
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn.functional as F
from common import ORIGINAL_MODEL, MODEL_DIR_PATH

class Similarity:
    def __init__(self):
        # Detect device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Load model to appropriate device
        self.model = SentenceTransformer(f'{MODEL_DIR_PATH}/{ORIGINAL_MODEL}', device=self.device)

    def sim_compute(self, query, demo):
        embedding1 = self.model.encode(query, show_progress_bar=False, device=self.device, convert_to_tensor=True)
        embedding2 = self.model.encode(demo, batch_size=8192, show_progress_bar=False, device=self.device, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)[0]
        return cosine_scores



