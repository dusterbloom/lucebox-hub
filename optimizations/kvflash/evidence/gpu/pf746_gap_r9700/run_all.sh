#!/bin/bash
# usage: run_all.sh TAG   (env as for run_arm.sh) runs arms A, B, C one after the other into out/TAG
cd ~/pf746
for a in A B C; do scripts/run_arm.sh $a $1 > out/$1.$a.driver.log 2>&1; echo "arm $a exit $?" >> out/$1.status; done
echo done >> out/$1.status
