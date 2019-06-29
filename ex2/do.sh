#!/bin/sh

DATAROOT=CAIA_backdoor.csv

# PDP, ALE
for i in 0 1 2; do
	./learn.py --dataroot "$DATAROOT" --method=rf --net runs/non-bd/*${i}_3.rfmodel --function pdp --fold $i
	./learn.py --dataroot "$DATAROOT" --method=rf --net runs/non-bd/*${i}_3.rfmodel --function ale --fold $i
	./learn.py --dataroot "$DATAROOT" --backdoor --method=rf --net runs/bd/*${i}_3.rfmodel --function pdp --fold $i
	./learn.py --dataroot "$DATAROOT" --backdoor --method=rf --net runs/bd/*${i}_3.rfmodel --function ale --fold $i
	
	./learn.py --dataroot "$DATAROOT" --method=nn --net runs/Jun28_17*${i}_3/*.pth --function pdp --fold $i
	./learn.py --dataroot "$DATAROOT" --method=nn --net runs/Jun28_17*${i}_3/*.pth --function ale --fold $i
	./learn.py --dataroot "$DATAROOT" --backdoor --method=nn --net runs/Jun28_19*${i}_3/*.pth --function pdp --fold $i
	./learn.py --dataroot "$DATAROOT" --backdoor --method=nn --net runs/Jun28_19*${i}_3/*.pth --function ale --fold $i
done

./plot.py pdp
./plot.py ale

# ICE
./learn.py --dataroot "$DATAROOT" --method=rf --net runs/non-bd/*0_3.rfmodel --function ice --fold 0
./learn.py --dataroot "$DATAROOT" --backdoor --method=rf --net runs/bd/*0_3.rfmodel --function ice --fold 0

./learn.py --dataroot "$DATAROOT" --method=nn --net runs/Jun28_17*0_3/*.pth --function ice --fold 0
./learn.py --dataroot "$DATAROOT" --backdoor --method=nn --net runs/Jun28_19*0_3/*.pth --function ice --fold 0


# Coefficients for surrogate model
for i in 0 1 2; do
	./learn.py --dataroot "$DATAROOT"  --method rf --net runs/non-bd/*${i}_3.rfmodel --function surrogate --fold $i
	./learn.py --dataroot "$DATAROOT"  --backdoor --method rf --net runs/bd/*${i}_3.rfmodel --function surrogate --fold $i
	./learn.py --dataroot "$DATAROOT"  --method nn --net runs/Jun28_17*${i}_3/*.pth --function surrogate --fold $i
	./learn.py --dataroot "$DATAROOT"  --backdoor --method nn --net runs/Jun28_19*${i}_3/*.pth --function surrogate --fold $i
done

# Performance benchmarks
for i in 0 1 2; do
	# unbackdoored model, unbackdoored data
	./learn.py --dataroot "$DATAROOT" --net runs/non-bd/Jun28_16-*${i}_3*.rfmodel --method rf --function test --fold $i
	./learn.py --dataroot "$DATAROOT" --net runs/Jun28_17-*${i}_3/*.pth --method nn --function test --fold $i
	
	# backdoored model, unbackdoored data
	./learn.py --dataroot "$DATAROOT" --net runs/bd/*${i}_3*.rfmodel --method rf --function test --fold $i --normalizationData ./CAIA_backdoor_backdoor_normalization_data.pickle
	./learn.py --dataroot "$DATAROOT" --net runs/Jun28_19-*${i}_3/*.pth --method nn --function test --fold $i --normalizationData ./CAIA_backdoor_backdoor_normalization_data.pickle
	
	# backdoor efficacy
	./learn.py --dataroot ./forward_backdoor.csv --net runs/bd/*${i}_3*.rfmodel --method rf --function test --fold $i --normalizationData ./CAIA_backdoor_backdoor_normalization_data.pickle
	./learn.py --dataroot ./backward_backdoor.csv --net runs/bd/*${i}_3*.rfmodel --method rf --function test --fold $i --normalizationData ./CAIA_backdoor_backdoor_normalization_data.pickle
	
	./learn.py --dataroot ./forward_backdoor.csv --net runs/Jun28_19-*${i}_3/*.pth --method nn --function test --fold $i --normalizationData ./CAIA_backdoor_backdoor_normalization_data.pickle
	./learn.py --dataroot ./backward_backdoor.csv --net runs/Jun28_19-*${i}_3/*.pth --method nn --function test --fold $i --normalizationData ./CAIA_backdoor_backdoor_normalization_data.pickle
done
