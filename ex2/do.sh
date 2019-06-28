#!/bin/sh

DATAROOT=CAIA_backdoor.csv

for i in `seq 0 2`; do
	./learn.py --dataroot "$DATAROOT" --method=rf --function pdp --fold $i
	./learn.py --dataroot "$DATAROOT" --method=rf --function ale --fold $i
	./learn.py --dataroot "$DATAROOT" --backdoor --method=rf --function pdp --fold $i
	./learn.py --dataroot "$DATAROOT" --backdoor --method=rf --function ale --fold $i
	
	#./learn.py --dataroot "$DATAROOT" --method=nn --net runs/Jun24*${i}_3/*.pth --function pdp --fold $i
	#./learn.py --dataroot "$DATAROOT" --method=nn --net runs/Jun24*${i}_3/*.pth --function ale --fold $i
	./learn.py --dataroot "$DATAROOT" --backdoor --method=nn --net runs/Jun26*${i}_3/*.pth --function pdp --fold $i
	./learn.py --dataroot "$DATAROOT" --backdoor --method=nn --net runs/Jun26*${i}_3/*.pth --function ale --fold $i
done

./learn.py --dataroot "$DATAROOT" --method=$method --function ice --fold 0
./learn.py --dataroot "$DATAROOT" --backdoor --method=$method --function ice --fold 0

./learn.py --dataroot "$DATAROOT" --method=nn --net runs/Jun24*0_3/*.pth --function ice --fold 0
./learn.py --dataroot "$DATAROOT" --backdoor --method=nn --net runs/Jun26*0_3/*.pth --function ice --fold 0

./plot.py pdp
./plot.py ale
