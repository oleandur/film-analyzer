import json
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image


class ShotSemanticSearch:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def build_index(
        self,
        analysis_path: str | Path,
        index_path: str | Path,
        batch_size: int = 16,
    ) -> Path:
        analysis_path = Path(analysis_path)
        index_path = Path(index_path)

        with analysis_path.open("r", encoding="utf-8") as file:
            analysis = json.load(file)

        shots = analysis.get("shots", [])
        if not shots:
            raise ValueError("No shots found in analysis file")

        shot_ids: list[int] = []
        image_tensors = []
        embeddings = []

        for shot in shots:
            thumbnail = analysis_path.parent / shot["thumbnail"]

            if not thumbnail.exists():
                raise FileNotFoundError(f"Thumbnail not found: {thumbnail}")

            with Image.open(thumbnail) as image:
                image_tensor = self.preprocess(image.convert("RGB"))

            shot_ids.append(int(shot["id"]))
            image_tensors.append(image_tensor)

            if len(image_tensors) == batch_size:
                embeddings.append(self._encode_image_batch(image_tensors))
                image_tensors = []

        if image_tensors:
            embeddings.append(self._encode_image_batch(image_tensors))

        embedding_matrix = np.concatenate(embeddings, axis=0)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            index_path,
            shot_ids=np.asarray(shot_ids, dtype=np.int32),
            embeddings=embedding_matrix,
        )

        return index_path

    def search(
        self,
        query: str,
        analysis_path: str | Path,
        index_path: str | Path,
        top_k: int = 5,
    ) -> list[dict]:
        analysis_path = Path(analysis_path)
        index_path = Path(index_path)

        with analysis_path.open("r", encoding="utf-8") as file:
            analysis = json.load(file)

        index = np.load(index_path)
        shot_ids = index["shot_ids"]
        image_embeddings = index["embeddings"]

        text_embedding = self._encode_text(query)
        scores = image_embeddings @ text_embedding

        top_k = min(top_k, len(scores))
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        shots_by_id = {
            int(shot["id"]): shot
            for shot in analysis.get("shots", [])
        }

        results = []

        for rank, index_position in enumerate(ranked_indices, start=1):
            shot_id = int(shot_ids[index_position])
            shot = shots_by_id[shot_id]

            results.append({
                "rank": rank,
                "shot_id": shot_id,
                "score": float(scores[index_position]),
                "start": shot["start"],
                "end": shot["end"],
                "thumbnail": shot["thumbnail"],
            })

        return results

    def _encode_image_batch(self, image_tensors: list[torch.Tensor]) -> np.ndarray:
        batch = torch.stack(image_tensors).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(batch, normalize=True)

        return features.cpu().numpy()

    def _encode_text(self, query: str) -> np.ndarray:
        tokens = self.tokenizer([query]).to(self.device)

        with torch.no_grad():
            features = self.model.encode_text(tokens, normalize=True)

        return features[0].cpu().numpy()
