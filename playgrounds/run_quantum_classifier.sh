tsp -S 16
for optim in rcd gd; do
    for lr in $(seq 0.05 0.01 0.2); do
        echo $lr $optim
        tsp python run_quantum_classifier.py --lr $lr --optim $optim
    done
done
