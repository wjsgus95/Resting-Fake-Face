
util_to_top_dir=".."
testset_dir="uniform_testset"

fake_dir=$util_to_top_dir/$testset_dir/"fake"
real_dir=$util_to_top_dir/$testset_dir/"real"

for img in $fake_dir/*; do
    echo "$(basename $img),0"
done

for img in $real_dir/*; do
    echo "$(basename $img),1"
done
