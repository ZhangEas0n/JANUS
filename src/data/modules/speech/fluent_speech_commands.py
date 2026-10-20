################################################################################
#
# Lightning data module for Fluent Speech Commands.
#
################################################################################

import csv
import pathlib

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset

from src.config_util import CastingConfig
from src.data.common import BatchDebugInfo, SpeechDataLoaderConfig
from src.data.modules.speech.speech_data_module import SpeechLightningDataModule
from src.data.modules.speech.training_batch_speech import (
    SpeechRecognitionDataSample,
    default_collate_fn,
)
from src.data.preprocess.base import Preprocessor
from src.data.util import load_raw_audio
from src.tokenizer.base import BaseTokenizer


@dataclass
class FluentSpeechCommandsDataModuleConfig(CastingConfig):
    dataset_folder: pathlib.Path
    train_csv: pathlib.Path
    val_csv: pathlib.Path
    test_csv: pathlib.Path
    train_collate_fn: str
    val_collate_fn: str
    test_collate_fn: str
    add_side_info: bool
    limit_samples: int


class FluentSpeechCommandsDataset(Dataset):
    def __init__(
        self,
        dataset_folder: pathlib.Path,
        csv_path: pathlib.Path,
        tokenizer: BaseTokenizer,
        pipeline: List[Preprocessor],
        intent_label_to_id: Dict[str, int],
        add_side_info: bool,
        limit_samples: int = -1,
    ):
        self.dataset_folder = pathlib.Path(dataset_folder)
        self.csv_path = pathlib.Path(csv_path)
        self.tokenizer = tokenizer
        self.pipeline = pipeline
        self.intent_label_to_id = intent_label_to_id
        self.add_side_info = add_side_info
        self.rows = self._load_rows()

        if limit_samples is not None and limit_samples > 0:
            self.rows = self.rows[:limit_samples]

    def _load_rows(self) -> List[dict]:
        with self.csv_path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def intent_label(row: dict) -> str:
        return "|".join(
            [
                row["action"].strip().lower(),
                row["object"].strip().lower(),
                row["location"].strip().lower(),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int) -> SpeechRecognitionDataSample:
        row = self.rows[index]
        audio_path = self.dataset_folder / row["path"]
        audio_sample = load_raw_audio(audio_path)
        transcription = row["transcription"].strip()
        transcription_int_sequence = self.tokenizer.encode_string(transcription)

        if len(transcription_int_sequence.shape) == 0:
            transcription_int_sequence = torch.stack([transcription_int_sequence])

        intent_label = self.intent_label(row)
        meta = {
            "speaker_id": row["speakerId"],
            "transcription": transcription,
            "action": row["action"].strip().lower(),
            "object": row["object"].strip().lower(),
            "location": row["location"].strip().lower(),
            "intent_label": intent_label,
            "intent_id": self.intent_label_to_id[intent_label],
        }

        sample = SpeechRecognitionDataSample(
            key=f"fsc-{index}",
            ground_truth=transcription_int_sequence,
            network_input=audio_sample,
            ground_truth_string=transcription,
            input_length=audio_sample.shape[1],
            ground_truth_sequence_length=transcription_int_sequence.shape[0],
            side_info=BatchDebugInfo(
                original_tensor=audio_sample,
                pipeline_progress=[],
                meta=meta,
            )
            if self.add_side_info
            else BatchDebugInfo(
                original_tensor=audio_sample,
                pipeline_progress=[],
                meta=meta,
            ),
        )

        for preprocessor in self.pipeline:
            processed = preprocessor.process(sample)
            if isinstance(processed, list):
                if len(processed) != 1:
                    raise ValueError("FSC speech pipeline must keep one sample per row")
                sample = processed[0]
            else:
                sample = processed

        return sample


class FluentSpeechCommandsDataModule(SpeechLightningDataModule):
    def __init__(
        self,
        cfg: FluentSpeechCommandsDataModuleConfig,
        dl_cfg: SpeechDataLoaderConfig,
        tokenizer: BaseTokenizer,
        train_pipeline: List[Preprocessor],
        val_pipeline: List[Preprocessor],
        test_pipeline: List[Preprocessor],
    ):
        super().__init__()
        self.cfg = cfg
        self.dl_cfg = dl_cfg
        self.train_pipeline = train_pipeline
        self.val_pipeline = val_pipeline
        self.test_pipeline = test_pipeline
        self._tokenizer = tokenizer
        self._vocabulary: List[str] = None
        self.intent_label_to_id: Dict[str, int] = {}
        self.intent_id_to_label: Dict[int, str] = {}
        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None

    def summary(self):
        print(
            "Fluent Speech Commands is ready: "
            f"train={len(self.train_ds)}, val={len(self.val_ds)}, "
            f"test={len(self.test_ds)}, intents={len(self.intent_label_to_id)}"
        )

    def vocabulary(self) -> List[str]:
        if self._vocabulary is None:
            self._vocabulary = list(self._tokenizer.vocabulary_dictionary().keys())
        return self._vocabulary

    def tokenizer(self) -> BaseTokenizer:
        return self._tokenizer

    def prepare_data(self) -> None:
        required_paths = [
            self.cfg.dataset_folder,
            self.cfg.train_csv,
            self.cfg.val_csv,
            self.cfg.test_csv,
        ]
        missing = [str(path) for path in required_paths if not pathlib.Path(path).exists()]
        if len(missing) > 0:
            raise FileNotFoundError(f"Missing FSC dataset paths: {missing}")

    def setup(self, stage: Optional[str] = None) -> None:
        self.intent_label_to_id = self._build_intent_vocabulary()
        self.intent_id_to_label = {
            value: key for key, value in self.intent_label_to_id.items()
        }
        self.train_ds = self._make_dataset(self.cfg.train_csv, self.train_pipeline)
        self.val_ds = self._make_dataset(self.cfg.val_csv, self.val_pipeline)
        self.test_ds = self._make_dataset(self.cfg.test_csv, self.test_pipeline)

    def _build_intent_vocabulary(self) -> Dict[str, int]:
        labels = set()
        for csv_path in [self.cfg.train_csv, self.cfg.val_csv, self.cfg.test_csv]:
            with pathlib.Path(csv_path).open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    labels.add(FluentSpeechCommandsDataset.intent_label(row))
        return {label: idx for idx, label in enumerate(sorted(labels))}

    def _make_dataset(
        self,
        csv_path: pathlib.Path,
        pipeline: List[Preprocessor],
    ) -> FluentSpeechCommandsDataset:
        return FluentSpeechCommandsDataset(
            dataset_folder=self.cfg.dataset_folder,
            csv_path=csv_path,
            tokenizer=self._tokenizer,
            pipeline=pipeline,
            intent_label_to_id=self.intent_label_to_id,
            add_side_info=self.cfg.add_side_info,
            limit_samples=self.cfg.limit_samples,
        )

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.dl_cfg.train_batch_size,
            collate_fn=self._determine_collate_fn(self.cfg.train_collate_fn),
            shuffle=True,
            num_workers=self.dl_cfg.num_workers,
            pin_memory=self.dl_cfg.pin_memory,
            drop_last=False,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            dataset=self.val_ds,
            batch_size=self.dl_cfg.val_batch_size,
            collate_fn=self._determine_collate_fn(self.cfg.val_collate_fn),
            shuffle=False,
            num_workers=self.dl_cfg.num_workers,
            pin_memory=self.dl_cfg.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            dataset=self.test_ds,
            batch_size=self.dl_cfg.test_batch_size,
            collate_fn=self._determine_collate_fn(self.cfg.test_collate_fn),
            shuffle=False,
            num_workers=self.dl_cfg.num_workers,
            pin_memory=self.dl_cfg.pin_memory,
            drop_last=False,
        )

    @staticmethod
    def _determine_collate_fn(name: str):
        if name == "default":
            return default_collate_fn
        raise ValueError(f"cannot determine collate_fn {name}")
