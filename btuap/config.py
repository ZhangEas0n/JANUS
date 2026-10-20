from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class BTUAPAttackConfig:
    ckpt_path: str = "model/wav2vec_base.ckpt"    #扰动路径
    output_path: str = "perturbations/original/SV_btuap_fsc_sr_optimized.pt"   #输出扰动路径
    evaluation_uap_path: Optional[str] = None      #评估扰动路径
    intent_head_path: str = "models/checkpoints/sr/intent_head_fsc.pt"     #分类头
    sample_rate: int = 16000   #采样率
    audio_length: float = 0.5       #音频长度，单位是s
    max_epochs: int = 2     
    steps_per_epoch: int = 500     
    tau_eval: float = 0.8     #SV阈值
    beta: float = 10.0        #没用了
    alpha: float = 0.5        #SV
    gamma: float = 0.5        #SR
    lambda_reg: float = 0.05  #不可感知性
    lambda_smooth: float = 0.01    #音频平滑度，可以用来优化感知性
    epsilon: float = 0.05     #snr范围
    pseudo_enrollment_samples: int = 50    #设定的注册用户
    sr_train_batch_size: int = 4           #分类头训练部分
    sr_train_max_num_samples: int = 16000 * 12
    intent_warmup_steps: int = 8000
    intent_head_lr: float = 3e-4
    sensitive_class_weight: float = 1.5
    intent_eval_interval_steps: int = 1000
    intent_eval_max_batches: Optional[int] = None
    intent_decision_threshold: float = 0.35
    intent_head_only: bool = False
    sr_loss_moving_average_window: int = 100
    sr_attack_eval_interval_steps: int = 100
    sr_attack_monitor_max_batches: Optional[int] = 50
    run_sr_eval_after_training: bool = True
    sr_eval_max_batches: Optional[int] = None
    run_sv_eval_after_training: bool = None
    sv_eval_max_batches: Optional[int] = None
    sensitive_class_ids: List[int] = field(default_factory=lambda: [1])
    sensitive_intent_labels: List[str] = field(
        default_factory=lambda: [
            "deactivate|lights|none",
            "deactivate|lights|kitchen",
            "deactivate|lights|bedroom",
            "deactivate|lights|washroom",
            "increase|heat|none",
            "increase|heat|kitchen",
            "increase|heat|bedroom",
            "increase|heat|washroom",
            "activate|lamp|none",
            "increase|volume|none",
        ]
    )
    sensitive_keywords: List[str] = field(
        default_factory=lambda: [
            "password",
            "passcode",
            "pin",
            "pincode",
            "code",
            "verification code",
            "security code",
            "one time password",
            "otp",
            "secret",
            "private",
            "confidential",
            "classified",
            "credential",
            "credentials",
            "username",
            "login",
            "sign in",
            "sign-in",
            "account",
            "profile",
            "identity",
            "identification",
            "passport",
            "driver license",
            "license number",
            "social security",
            "ssn",
            "birth date",
            "birthday",
            "address",
            "phone number",
            "telephone number",
            "email",
            "email address",
            "bank",
            "bank account",
            "account number",
            "routing number",
            "transaction",
            "account",
            "transfer",
            "money",
            "payment",
            "pay",
            "paid",
            "invoice",
            "bill",
            "purchase",
            "order",
            "checkout",
            "credit",
            "debit",
            "credit card",
            "debit card",
            "card number",
            "cvv",
            "expiration date",
            "wallet",
            "balance",
            "cash",
            "deposit",
            "withdraw",
            "wire transfer",
            "loan",
            "tax",
            "refund",
            "insurance",
            "medical record",
            "health record",
            "diagnosis",
            "prescription",
            "patient",
            "secret",
            "private",
            "unlock",
            "lock",
            "open",
            "access",
            "grant access",
            "authorized",
            "permission",
            "approve",
            "confirm",
            "verify",
            "verification",
            "authenticate",
            "authentication",
            "biometric",
            "fingerprint",
            "face id",
            "voiceprint",
            "surveillance",
            "camera",
            "microphone",
            "location",
            "gps",
            "track",
            "tracking",
            "monitor",
            "security",
            "safe",
            "alarm",
            "emergency",
            "warning",
            "danger",
            "threat",
            "attack",
            "weapon",
            "gun",
            "bomb",
            "explosive",
        ]
    )

    @property
    def audio_length_samples(self) -> int:
        if self.audio_length <= 0:
            raise ValueError(f"audio_length must be positive, got {self.audio_length}")
        if self.audio_length <= 10:
            return max(1, int(round(self.audio_length * self.sample_rate)))
        return int(round(self.audio_length))

    def ensure_output_parent(self):
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

    def ensure_intent_head_parent(self):
        Path(self.intent_head_path).parent.mkdir(parents=True, exist_ok=True)

    def resolve_paths(self, project_root):
        project_root = Path(project_root)
        for field_name in ["ckpt_path", "output_path", "evaluation_uap_path", "intent_head_path"]:
            value = getattr(self, field_name)
            if value is None:
                continue
            path = Path(value)
            if not path.is_absolute():
                setattr(self, field_name, str(project_root / path))
