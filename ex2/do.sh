#!/bin/sh

DATAROOT=CAIA_backdoor.csv

for i in `seq 0 2`; do
	./learn.py --dataroot "$DATAROOT" --method=rf --net runs/non-bd/*${i}_3.rfmodel --function pdp --fold $i
	./learn.py --dataroot "$DATAROOT" --method=rf --net runs/non-bd/*${i}_3.rfmodel --function ale --fold $i
	./learn.py --dataroot "$DATAROOT" --backdoor --method=rf --net runs/bd/*${i}_3.rfmodel --function pdp --fold $i
	./learn.py --dataroot "$DATAROOT" --backdoor --method=rf --net runs/bd/*${i}_3.rfmodel --function ale --fold $i
	
	./learn.py --dataroot "$DATAROOT" --method=nn --net runs/Jun28_17*${i}_3/*.pth --function pdp --fold $i
	./learn.py --dataroot "$DATAROOT" --method=nn --net runs/Jun28_17*${i}_3/*.pth --function ale --fold $i
	./learn.py --dataroot "$DATAROOT" --backdoor --method=nn --net runs/Jun28_19*${i}_3/*.pth --function pdp --fold $i
	./learn.py --dataroot "$DATAROOT" --backdoor --method=nn --net runs/Jun28_19*${i}_3/*.pth --function ale --fold $i
done

./learn.py --dataroot "$DATAROOT" --method=rf --net runs/non-bd/*0_3.rfmodel --function ice --fold 0
./learn.py --dataroot "$DATAROOT" --backdoor --method=rf --net runs/bd/*0_3.rfmodel --function ice --fold 0

./learn.py --dataroot "$DATAROOT" --method=nn --net runs/Jun28_17*0_3/*.pth --function ice --fold 0
./learn.py --dataroot "$DATAROOT" --backdoor --method=nn --net runs/Jun28_19*0_3/*.pth --function ice --fold 0

./plot.py pdp
./plot.py ale
