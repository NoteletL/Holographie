#!/bin/bash

sessName=$1
echo $sessName


for i in $(seq 1 25);
do
	echo "img$i" > file_std$i.res
	grep "img$i " $sessName | cut -d" " -f5 | tr "." "," >> file_std$i.res 
    echo "img$i" > file_psnr$i.res
	grep "img$i " $sessName | cut -d" " -f3 | cut -d, -f1 |  tr "." "," >> file_psnr$i.res 
done

paste -d ";" file_std*.res > STD.res
paste -d ";" file_psnr*.res > PSNR.res
rm file*.res
