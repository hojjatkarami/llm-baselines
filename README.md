# LLM-baselines

A brief introduction of `Artemis` team participated for LLM track in LauzHack 2024.

# Applied Changes

- We used `llama2` model as it was marginally better than the `GPTbase` model.
- To be able to increase the `batch_size` to 96, we reduced the `sequence_length` to 384, `n_layer` and `n_head` to 4.
- We also tried different learning rate schedulers and decided to change it to `torch.optim.lr_scheduler.CyclicLR`:
    ```python
    scheduler = torch.optim.lr_scheduler.CyclicLR(
                    opt,
                    0.1 * args.lr,
                    args.lr,
                    step_size_up=2000,
                    step_size_down=None,
                    mode="exp_range",
                    gamma=1.0,
                    scale_fn=None,
                    scale_mode="cycle",
                    cycle_momentum=False,
                    base_momentum=0.8,
                    max_momentum=0.9,
                    last_epoch=-1,
                )
    ```

The final command to run on `NVIDIA A100-SXM4-40GB`:
```sh
python ./src/main.py --config_format base --wandb --wandb_project LLM --n_layer 4 --n_head 4 --model llama2 --seed 123 --batch_size 96  --sequence_length 384
```
link to wandb report.
Model wieghts can be downloaded from [here]().

We have also explored some other ideas, which were not effective in our case:
- We tried using MoE framework, however, it seems that the model was too large for the given resources.
- We used `GradScaler` from `torch.cuda.amp`, but it lead to nan loss in the middle of training
- We tried to implemet different attention implementations such as `FlashAttention-v2`, `Sparse Attention` but they were not effective in our case.