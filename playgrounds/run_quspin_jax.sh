# for i in 5 10 15 20 25 30; do
#     echo "Run $i"
#     python test_quspin_jax.py --p $i
# done


# run the experiments with noise
# for i in 5 10 15 20 25 30; do
#     for sigma in 0.001 0.005 0.01 0.05 0.1; do
#         echo "Run $i with noise $sigma"
#         python test_quspin_jax_noise.py --p $i --sigma $sigma
#     done
# done


# run the experiments with noise
# for i in 5 10 15 20 25 30; do
#     for sigma in 0.001 0.005 0.01 0.05 0.1; do
#         echo "Run $i with noise $sigma"
#         python test_quspin_jax_noise_heisenberg.py --p $i --sigma $sigma
#     done
# done


# for i in 10 15 20 25 30; do
#     for sigma in 0.1; do
#         echo "Run $i with noise $sigma"
#         python test_quspin_jax_noise_heisenberg.py --p $i --sigma $sigma --repeat 1
#     done
# done


# for i in 10 15 20 25 30; do
#     for sigma in 0.1; do
#         echo "Run $i with noise $sigma"
#         python test_quspin_jax_noise_new.py --p $i --sigma $sigma --repeat 1
#     done
# done


# for i in 10 15 20 25 30; do
#     for sigma in 0.1; do
#         for lr in $(seq 0.001 0.001 0.05); do
#             echo "Run $i with noise $sigma"
#             python test_quspin_jax_noise_heisenberg.py --p $i --sigma $sigma --repeat 1 --lr_gd $lr --lr_rcd $lr
#         done  
#     done
# done



for i in 10 15 20 25 30; do
    for sigma in 0.1; do
        for lr in $(seq 0.001 0.001 0.05); do
            echo "Run $i with noise $sigma"
            python test_quspin_jax_noise_new.py --p $i --sigma $sigma --repeat 1 --lr_gd $lr --lr_rcd $lr
        done
    done
done