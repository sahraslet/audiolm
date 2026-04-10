# train.ps1
# PowerShell version of the original train.sh


# Build argument list
$argsList = @(
  "--tokenizer_path"; "Qwen/Qwen2.5-0.5B"
  "--dataset_path"; "sahara22/audiolm-eval"
  "--checkpoint_dir"; ".\audiolm\checkpoints_scheduler"
  "--logfile_path"; ".\audiolm\logs\train_scheduler.log"
  "--wandb_project_name"; "audio-lm"
  "--wandb_entity"; "sahratls-universit-t-des-saarlandes-saarland-university"
  "--wandb_run_name"; "first-run"
  "--lr"; "1e-5"
  "--num_epochs"; "2"
  "--batch_size"; "1"
  "--eval_every"; "8000"
  "--save_every"; "8000"
  "--grad_accumulation_steps"; "16"
  "--device"; "cuda"
  "--push_to_hub";
  "--hub_repo_id"; "sahara22/audio_language_model"
)

# Run Python script with arguments
python C:\Users\Raschied\git\audiolm\scripts\train.py @argsList

# To capture output to the logfile as well as console, uncomment:
# python @argsList 2>&1 | Tee-Object -FilePath "C:\data\users\sslet\audiolm\logs\train.log"
