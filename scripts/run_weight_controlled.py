from simple_slurm import Slurm
import subprocess


def run(lmax, shlmax, ns, nv):
    run_name = f'WC_lmax{lmax}_shlmax{shlmax}_nv{nv}_ns{ns}'

    cmd = f'python -m train --run_name {run_name} --test_sigma_intervals --esm_embeddings_path data/esm2_3billion_embeddings_P00918_split.pt --log_dir workdir --lr 1e-3 --tr_sigma_min 0.1 --tr_sigma_max 19 --rot_sigma_min 0.03 --rot_sigma_max 1.55 --batch_size 16 --ns {ns} --nv {nv} --num_conv_layers 5 --dynamic_max_cross --scheduler plateau --scale_by_sigma --dropout 0.1 --remove_hs --c_alpha_max_neighbors 24 --receptor_radius 15 --num_dataloader_workers 1 --cudnn_benchmark --val_inference_freq 5 --num_inference_complexes 500 --use_ema --scheduler_patience 30 --n_epochs 850 --wandb --split_train data/P00918_split/P00918_train --split_val data/P00918_split/P00918_val --split_test data/P00918_split/P00918_test --use_order_repr {lmax} --use_sh_lmax {shlmax}'

    slurm = Slurm(
        c=8, 
        job_name=run_name,
        t='2-00:00', 
        p='gpu_quad', 
        out=f'out/{run_name}.%x.%j.out',
        err=f'out/{run_name}.%x.%j.out',
        gres='gpu:1', 
        mem='32G'
    )

    print(f'Executing sbatch: \n{slurm}\n{cmd}')
    slurm.sbatch(cmd)

    # subprocess.run(cmd, shell=True, check=True) 
    return cmd


# Model sizes obtained using large model setup with ns=48, nv=10
numel_lg = {
    0: {
        'numel_embedding': 121248,
        'numel_conv': 8895076,
        'numel_final_layer': 11090,
    },
    1: {
        'numel_embedding': 121248,
        'numel_conv': 20115876,
        'numel_final_layer': 11090,
    },
    2: {
        'numel_embedding': 121248,
        'numel_conv': 29206876,
        'numel_final_layer': 11090,
    },
    3: {
        'numel_embedding': 121248,
        'numel_conv': 34187596,
        'numel_final_layer': 11090,
    }
}
# Model sizes obtained using small model setup with ns=24, nv=6
numel_sm = {
    0: {
        'numel_embedding': 48720,
        'numel_conv': 999124,
        'numel_final_layer': 2858,
    },
    1: {
        'numel_embedding': 48720,
        'numel_conv': 2235412,
        'numel_final_layer': 2858,
    },
    2: {
        'numel_embedding': 48720,
        'numel_conv': 3404812,
        'numel_final_layer': 2858,
    },
    3: {
        'numel_embedding': 48720,
        'numel_conv': 4110508,
        'numel_final_layer': 2858,
    }
}

# Model sizes obtained using small model setup with incorrect weight controlled (assumed linear relationship between ns, nv and model size)
numel_sm_wc_incorrect = {
    0: {
        'ns': 20,
        'nv': 4,
        'numel_embedding': 40120,
        'numel_conv': 593404,
        'numel_final_layer': 2222,
    },
    1: {
        'ns': 9,
        'nv': 2,
        'numel_embedding': 17460,
        'numel_conv': 125617,
        'numel_final_layer': 803,
    },
    2: {
        'ns': 6,
        'nv': 1,
        'numel_embedding': 11532,
        'numel_conv': 48064,
        'numel_final_layer': 500,
    },
    3: {
        'ns': 5,
        'nv': 1,
        'numel_embedding': 9580,
        'numel_conv': 36613,
        'numel_final_layer': 407,
    }
}


# Model sizes obtained using small model setup with correct empirical weight controlled ns, nv
numel_sm_wc = {
    0: {
        'ns': 24,
        'nv': 6,
        'numel_embedding': 17400,
        'numel_conv': 999124,
        'numel_final_layer': 2858,
    },
    1: {
        'ns': 18,
        'nv': 5,
        'numel_embedding': 12510,
        'numel_conv': 1016860,
        'numel_final_layer': 1928,
    },
    2: {
        'ns': 16,
        'nv': 4,
        'numel_embedding': 10960,
        'numel_conv': 1034804,
        'numel_final_layer': 1650,
    },
    3: {
        'ns': 14,
        'nv': 4,
        'numel_embedding': 9450,
        'numel_conv': 966616,
        'numel_final_layer': 1388,
    }
}

# for order in [0, 1, 2, 3]:
#     # Scale calculated using model sizes above
#     scale = numel_sm[order]['numel_conv'] / numel_sm[0]['numel_conv']
#     ns = 20 / scale
#     nv = 4 / scale
#     print('ns: ', round(ns), ns)
#     print('nv:', round(nv), nv)
#     run(order, round(ns), round(nv))

# for order in [0, 1, 2, 3]:
for order in [2, 3]:
    run(order, order+1, numel_sm_wc[order]['ns'], numel_sm_wc[order]['nv'])

# order = 0
# run(order, numel_sm_wc[order]['ns'], numel_sm_wc[order]['nv'])
