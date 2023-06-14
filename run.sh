# If second argument is provided store in seed variable
if [ -z "$2" ]
then
    seed=17
else
    seed=$2
fi
echo ./configs/$1
bash ./configs/$1.sh $seed