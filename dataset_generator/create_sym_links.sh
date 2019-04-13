split1='/dev/shm/split1/'
lim1=10

split2='/dev/dataset/split2/'
lim2=20

count=1
for i in $(seq 10)
do
	link1=$split1$i"_0.npy"
	ln -s $link1 $count".npy"
	count=$(expr $count + 1)
	link2=$split1$i"_1.npy"
	ln -s $link2 $count".npy"
	count=$(expr $count + 1)
done


for i in $(seq 11 20)
do
        link1=$split2$i"_0.npy"
        ln -s $link1 $count".npy"
        count=$(expr $count + 1)
        link2=$split2$i"_1.npy"
        ln -s $link2 $count".npy"
        count=$(expr $count + 1)
done
