import argparse

import jax
import jax.numpy as jnp
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import wandb
from numpyro import distributions as dist

from qwen_fineTuning_PtoC import load_text_classification_data
from qwen_fineTuning_PtoC_fixCps import setup_metrics
from src.utils import calibration_error as ece_fn, init_model, load_checkpoint_new, load_fn


def main(sweep_id: str, use_art: bool):

    api = wandb.Api()

    t_bar = tqdm(api.sweep(f"ddold/qwen_PtoC/{sweep_id}").runs)
    t_bar.set_description(f"Sweep id: {sweep_id}")

    for run in t_bar:
        config = deepcopy(run.config)
        
        if not use_art:
            # config e.g.: "ddold/qwen_PtoC/winogrande_m_dataset:v0"
            config['dataset_path'] = f"artifacts/{config['dataset_path'].split('/')[-1]}/test_data.npz"
            config['net_kwargs']['model_path'] = f"artifacts/{config['net_kwargs']['model_path'].split('/')[-1]}"
        config['curve_params']['k'] = 0

        # Load data inside the loop so dataset can vary per run if needed
        test_input_ids, test_attention_mask, test_labels, test_target_ids, test_unique_target_ids = \
            load_text_classification_data(config['dataset_path'], run=None, test=True)

        s_model, params = init_model(config, test_unique_target_ids)
        params, bma_weights, trainable_params, init_trainable_params, pretrained_params, last_trainable_params = \
            load_checkpoint_new(run, params, s_model)

        metric_fn = setup_metrics(
            s_model, 8, test_input_ids, test_attention_mask, test_labels,
            1, use_linspace=True, average=False,
        )

        all_logits = []
        all_metrics = []
        for p_param in pretrained_params:
            if p_param:
                params['params'] = jax.tree.map(load_fn, s_model.train_mask, params['params'], p_param)
                params = s_model.set_lora_rho(rho_s=0., rho_w=0., params=params)
                metrics, logits = metric_fn(jax.random.PRNGKey(0), params)
                all_logits.append(logits)
                all_metrics.append(metrics)

        logits_test = jnp.stack(all_logits, axis=0).squeeze()  # (n_ensembles, n_data, output_dim)
        print(f"Shape stacked logits {logits_test.shape}")

        # Bayesian model averaging
        log_probs = jax.nn.log_softmax(logits_test, axis=-1)
        post_predict_logits = jax.nn.logsumexp(log_probs, axis=0) - jnp.log(log_probs.shape[0])

        preds = jnp.argmax(post_predict_logits, axis=-1)
        accuracy = jnp.mean(preds == test_labels)
        ece = ece_fn(post_predict_logits, test_labels, num_bins=15)
        ll = dist.Categorical(logits=post_predict_logits).log_prob(test_labels).mean()

        print("\033[94m" + f"Post-prediction performance: Accuracy={accuracy:.4f}, ECE={ece:.4f}, NLL={-ll:.4f}" + "\033[0m")
        run.summary['de_post_pred_acc'] = accuracy
        run.summary['de_post_pred_ece'] = ece
        run.summary['de_post_pred_ll'] = ll
        run.summary.update()

        del params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate deep-ensemble performance for a wandb sweep.")
    parser.add_argument("--sweep-id", type=str, required=True, help="Wandb sweep id (e.g. 3p3okc1w)")
    parser.add_argument("--use-art", action="store_true", help="Load data from wandb artifact storage. If not set, data will be loaded from local 'artifacts' folder with ds and model specified in config.")
    args = parser.parse_args()

    main(
        sweep_id=args.sweep_id,
        use_art=args.use_art,
    )
