for N in $(seq 4 2 8); do
    python generate_heisenberg_ham.py --N $N
done