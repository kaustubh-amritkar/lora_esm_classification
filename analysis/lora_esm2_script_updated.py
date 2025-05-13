import os
import torch
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))
from transformers import AutoModelForTokenClassification, AutoTokenizer, PreTrainedModel, EsmConfig, EsmPreTrainedModel
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
# from tomlkit import value
import torch
import esm
import pandas as pd
import numpy as np
import random
import os
import wandb
import pickle as pkl
from datetime import datetime
from sklearn.metrics import roc_auc_score
import accelerate
from accelerate import Accelerator
from huggingface_hub import notebook_login
from torch.utils.data import Dataset, random_split
from transformers import (
    EsmForTokenClassification,
    EsmForMaskedLM,
    EsmModel,
    EsmTokenizer,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import gc
import copy

def train_protein_model():
    
    def wandb_hp_space():
        return {
            "method": "random",
            "metric": {"name": "accuracy", "goal": "maximize"},
            "parameters": {
                "learning_rate": {"distribution": "uniform", "min": 1e-5, "max": 1e-3},
                # "per_device_train_batch_size": {"values": [2, 4, 8, 16]},
                "per_device_train_batch_size": {"values": [2]},
            },
        }
    
    accelerator = Accelerator()

    large_dataset = pd.read_pickle('/home/kaustubh/RuBisCO_ML/ESM_LoRA/data/large_assay_dataset_wco2_wiptg_bimodal_activity_threshold.pkl')
    large_dataset = large_dataset[["LSU_id", "SSU_id", "lsu_seq", "ssu_seq", "activity_binary"]]

    mutant_dataset = pd.read_pickle("/home/kaustubh/RuBisCO_ML/ESM_LoRA/data/form_III_IB_anc_variants_binary_activity_only_wrt_inactive_variant.pkl")
    mutant_dataset.rename(columns={"activity_binary_wrt_inactive_Anc": "activity_binary"}, inplace=True)

    combined_dataset = pd.concat([large_dataset, mutant_dataset], ignore_index=True)
    combined_dataset = combined_dataset[combined_dataset['LSU_id'] != "SUMO"]
    combined_dataset = combined_dataset[combined_dataset['LSU_id'] != "DEAD"]
    combined_dataset['LSU_SSU_id'] = combined_dataset.apply(lambda x: x['LSU_id'] + "-" + x['SSU_id'] if pd.notna(x['SSU_id']) else x['LSU_id'] + "-none", axis=1)
    combined_dataset['LSU_SSU_seq'] = combined_dataset.apply(lambda x: x['lsu_seq'] + x['ssu_seq'] if (pd.notna(x['ssu_seq']) and x["SSU_id"] != "SUMO") else x['lsu_seq'], axis=1)
    combined_dataset_large_subet = combined_dataset[~combined_dataset['LSU_id'].str.startswith("Anc")]
    combined_dataset_mutant_subet = combined_dataset[combined_dataset['LSU_id'].str.startswith("Anc")]
    
    ### Training/Testing on parts of the whole dataset ###
    sequences = combined_dataset['LSU_SSU_seq'].to_list()
    binary_activity = combined_dataset['activity_binary'].to_list()
    lsu_ssu_ids = combined_dataset['LSU_SSU_id'].to_list()

    # concat_all_exp_data = pd.read_pickle('/home/kaustubh/RuBisCO_ML/ESM_LoRA/data/processed_combined_all_exp_assays_data.pkl')
    
    # formIII_lsu_variant_data_df = concat_all_exp_data.query('LSU_id.str.startswith("Anc393") or LSU_id.str.startswith("Anc367") or LSU_id == "Anc365" or LSU_id == "Anc366"')
    # formIII_lsu_variant_data_df['fixed_threshold_activity'] = formIII_lsu_variant_data_df['mean_reading'].apply(lambda x: 1 if x >= 50 else 0)

    # sequences = formIII_lsu_variant_data_df['lsussu_seq'].to_list()
    # binary_activity = formIII_lsu_variant_data_df['fixed_threshold_activity'].to_list()
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    class ProteinDataset(Dataset):
        def __init__(self, sequences, binary_activity, tokenizer, max_length=1024):
            self.sequences = sequences
            self.binary_activity = binary_activity
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            sequence = self.sequences[idx][:self.max_length]
            binding_site = self.binary_activity[idx]
            encoding = self.tokenizer(sequence, truncation=True, padding='max_length', max_length=self.max_length)
            encoding['labels'] = binding_site # + [-100] * (self.max_length - len(binding_site))  # Ignore extra padding tokens
            return encoding
    
    ## Making sure to split the dataset based on LSU_ids
    def generate_train_test(Xs, ys, headers, random_seed, fraction=0.8):
        lsu_list = [x.split("-")[0] for x in headers]
        lsu_list_uniq = pd.Series(lsu_list).unique().copy()
        random.seed(random_seed)
        random.shuffle(lsu_list_uniq)
        train_size = int(len(lsu_list_uniq) * fraction)
        train_set = lsu_list_uniq[:train_size]
        test_set = lsu_list_uniq[train_size:]
        train_indices = [i for i, x in enumerate(lsu_list) if x in train_set]
        test_indices = [i for i, x in enumerate(lsu_list) if x in test_set]

        Xs_train = [Xs[i] for i in train_indices]
        Xs_test = [Xs[i] for i in test_indices]
        ys_train = [ys[i] for i in train_indices]
        ys_test = [ys[i] for i in test_indices]
        return train_indices, test_indices, train_set, test_set, Xs_train, Xs_test, ys_train, ys_test

    train_indices, test_indices, train_ids, test_ids, Xs_train, Xs_test, ys_train, ys_test = generate_train_test(sequences, binary_activity, lsu_ssu_ids, 42)

    train_dataset = ProteinDataset(Xs_train, ys_train, tokenizer)
    val_dataset = ProteinDataset(Xs_test, ys_test, tokenizer)
    train_dataset, val_dataset = accelerator.prepare(train_dataset, val_dataset)
    
    class CustomModel(EsmPreTrainedModel):
        def __init__(self, model=None):
            print(torch.cuda.memory_stats())
            super().__init__(EsmConfig.from_pretrained("facebook/esm2_t33_650M_UR50D"))
            self.backbone = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

            self.outputs = torch.nn.Linear(1280, 1)

        def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            labels=None,
            inputs_embeds=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        ):
            outputs = self.backbone(
                input_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

            sequence_output = outputs.last_hidden_state # (B, L, 1280)

            ## Following for getting the BOS token embedding
            # bos_emb = sequence_output[:,0] # (B, L, 1280) -> (B, 1280)
            # # outputs = [self.outputs[i](sequence_output) for i in range(5)]
            # outputs = self.outputs(bos_emb).squeeze(1) # (B,)

            ## Following for getting the mean of the sequence embeddings
            mask_sum = attention_mask.sum(dim=1, keepdim=True).float()
            mean_emb = (sequence_output * attention_mask.unsqueeze(-1)).sum(dim=1) / mask_sum
            outputs = self.outputs(mean_emb).squeeze(1) # (B,)

            # if labels, then we are training
            loss = None
            if labels is not None:
                assert outputs.shape == labels.shape, f"{outputs}, {labels}"
                loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
                loss = loss_fn(outputs, labels.float())
                # loss = sum(losses)/len(losses)

            return {
                "loss": loss,
                # "last_hidden_state": sequence_output,
                "logits": outputs
            }

        def get_embedding(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            labels=None,
            inputs_embeds=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        ):
            outputs = self.backbone(
                input_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

            return outputs
        
    def model_init():
        gc.collect()
        torch.cuda.empty_cache()
        torch.manual_seed(42)
        base_model = CustomModel()
        config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=16,
            lora_alpha=16,
            target_modules=["query", "key", "value"],  # Apply LoRa to self-attention layers
            lora_dropout=0.1,
            bias="all",
        )
        lora_model = get_peft_model(base_model, config)
        return accelerator.prepare(lora_model)
    
    def cleanup_model(trainer):
        del trainer.model
        torch.cuda.empty_cache()
        gc.collect()
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred # (B,)
        # print("Predictions shape:", np.shape(predictions))
        # print("Labels shape:", np.shape(labels))
        # print(predictions.shape)
        # predictions = np.argmax(predictions, axis=2)  # Convert logits to class labels
        # labels = accelerator.gather(labels)
        # mask = labels != -100
        # accuracy = (predictions[mask] == labels[mask]).mean()
        accuracy = roc_auc_score(labels, predictions)
        return {'accuracy': accuracy}
    
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"/home/kaustubh/RuBisCO_ML/ESM_LoRA/training_runs/esm2_t33_650M-finetuned-lora_{timestamp_str}"

    args = TrainingArguments(
            output_dir,
            evaluation_strategy="epoch",
            learning_rate=5e-3,
            per_device_train_batch_size=1,
            num_train_epochs=5,
            logging_steps=10,
            load_best_model_at_end=False,
            metric_for_best_model="accuracy",
            save_strategy="epoch",
            label_names=["labels"],
            ddp_find_unused_parameters=False,
            save_total_limit=1,
        )
    
    trainer = Trainer(
        model=None,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        model_init=model_init,
    )

    # best_trial = trainer.hyperparameter_search(
    #     direction="maximize",
    #     backend="wandb",
    #     hp_space=wandb_hp_space,
    #     n_trials=10,
    # )

    def hyperparameter_search(trainer, hp_space, n_trials):
        best_trial_metric = None
        lr_min, lr_max = hp_space["learning_rate"]["min"], hp_space["learning_rate"]["max"]
        per_device_train_batch_size_values = hp_space["per_device_train_batch_size"]["values"]
        lr_range = lr_max - lr_min
        lr_step = lr_range / n_trials
        for i in range(n_trials):
            print(f"Trial {i+1}/{n_trials}")
            # hyperparameters = {
            #     k: v for k, v in zip(hp_space.keys(), [hp_space[k]["values"][random.randint(0, len(hp_space[k]["values"])-1)] for k in hp_space.keys()])
            # }
            # hyperparameters["learning_rate"] = hp_space["learning_rate"]["distribution"]
            # Sample lr
            lr = lr_min + lr_step * i
            print('learning_rate: ', lr)
            trainer.args.learning_rate = lr
            trainer.args.per_device_train_batch_size = per_device_train_batch_size_values[0]
            trainer.train()
            print('trainer:', trainer)
            print("Best current Metric:", trainer.state.best_metric)
            if best_trial_metric is not None:
                print("Best trial metric:", best_trial_metric)

            if best_trial_metric is None or trainer.state.best_metric > best_trial_metric:
                print("Updating best trial")
                best_trial_metric = trainer.state.best_metric
                best_trial = {
                    "learning_rate": lr,
                    "per_device_train_batch_size": trainer.args.per_device_train_batch_size,
                    best_trial_metric: trainer.state.best_metric
                }
            # Delete the model to save memory
            # cleanup_model(trainer)
        return best_trial
    
    hp_space = wandb_hp_space()
    print(hp_space.keys())
    
    best_trial = hyperparameter_search(trainer, wandb_hp_space()["parameters"], 10)

    print("Best Trial:", best_trial)

    def train_final_model(best_trial):
        # best_hyperparameters = best_trial.hyperparameters   ## If using the the hyperparameter_search from hugging face
        model = model_init()

        # args.learning_rate = best_hyperparameters["learning_rate"] ## If using the the hyperparameter_search from hugging face## If using the the hyperparameter_search from hugging face
        # args.per_device_train_batch_size = best_hyperparameters["per_device_train_batch_size"]  ## If using the the hyperparameter_search from hugging face
        args.learning_rate = best_trial["learning_rate"]    ## If using the the hyperparameter search function
        args.per_device_train_batch_size = best_trial["per_device_train_batch_size"]    ## If using the the hyperparameter search function
        final_trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        final_trainer.train()
        # Explicitly save the model's configuration
        model.config.save_pretrained(output_dir)
        # Save the model
        final_trainer.save_model(output_dir)

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    train_final_model(best_trial)