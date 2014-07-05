
src="$1"
dst="$2"
size="$3"

pid=$$
base_name="/tmp/base_${pid}"
iconvert "${src}" ${base_name}.tif


for res in 2048 1024 512 256 128 64 32 16 8 4 2
do
    if [ $res -gt $size ]
    then
        continue
    fi
    oiiotool ${src} --resize ${res}x0 -o ${base_name}_${res}.tif
done

bash -x ./build_mipmap.sh "${base_name}_" "${dst}"
