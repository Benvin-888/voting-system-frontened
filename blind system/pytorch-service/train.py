import torchaudio
from datasets import Dataset, Audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Trainer, TrainingArguments
import pandas as pd
import torch

# Load CSV (columns: audio_path, transcription)
df = pd.read_csv('voting_phrases.csv')
dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")

def prepare_dataset(batch):
    audio = batch["audio"]
    # batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["labels"] = processor(text=batch["transcription"]).input_ids
    return batch

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

train_test = dataset.train_test_split(test_size=0.1)
train_dataset = train_test["train"]
eval_dataset = train_test["test"]

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_steps=500,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor,
)

trainer.train()
model.save_pretrained("./asr_model")
processor.save_pretrained("./asr_model")